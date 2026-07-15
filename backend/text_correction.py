"""
LLM Text Correction (Phase 3B).

Post-Processing des transkribierten Textes mit einem lokalen Qwen3.5-0.8B-
Modell (ONNX, q4) via *plain* onnxruntime — ohne onnxruntime-genai, ohne
genai-Format, ohne torch. Das optimum-q4-Export laeuft direkt auf der
onnxruntime CPU, die ohnehin schon (fuer Parakeet) installiert ist.

Architektur: Qwen3.5 ist ein GatedDeltaNet-Hybrid-Modell (lineare Attention
+ volle Attention). Wir nutzen **Full-Recompute-Greedy-Decode**: jeder
Token-Schritt verarbeitet die komplette (wachsende) Sequenz mit geleertem
Cache neu. Das ist korrekt und simpel (kein Hybrid-Cache-Management noetig),
aber O(n^2). Inkrementeller KV-Cache ist eine spaetere Optimierung, falls
die CPU-Inferenz zu langsam sein sollte.

Robustheit: bei JEDEM Fehler (Modell fehlt, ONNX-Op, Speicher) wird der
Originaltext unveraendert zurueckgegeben — die Transkriptions-Pipeline
stuerzt niemals wegen der Korrektur ab.
"""

import logging
import os
import re
import threading
import time
from pathlib import Path

import numpy as np

from runtime_hooks.path_redirect import MODELS_DIR

logger = logging.getLogger(__name__)

# Qwen3.5 Special Tokens (siehe tokenizer_config.json / config.json)
IM_START = 248045
IM_END = 248046
EOS_TOKEN_ID = 248044

DEFAULT_MODEL_DIR = str(MODELS_DIR / "qwen3.5-0.8b-onnx")
DEFAULT_QUANT = "q4"

_SYSTEM_PROMPT = (
    "Du bist ein Korrekturassistent fuer transkribierte deutsche Diktate. "
    "Korrigiere den Text: entferne Fuellwoerter und Disfluenzen (aehm, hm, ...), "
    "hebe Grammatik und Rechtschreibung, loese Selbstkorrekturen auf. Behalte "
    "Sinn, Stil und die urspruengliche Wortstellung bei. Gib NUR den "
    "korrigierten Text zurueck, ohne Erklaerung, ohne Anfuehrungszeichen."
)


class TextCorrector:
    """Lazy-loaded Qwen3.5-0.8B Korrektur-Engine (plain onnxruntime, CPU)."""

    def __init__(self, model_dir: str = DEFAULT_MODEL_DIR, quant: str = DEFAULT_QUANT):
        self.model_dir = model_dir
        self.quant = quant
        self._lock = threading.Lock()
        self._loaded = False
        self._embed = None
        self._decoder = None
        self._tokenizer = None
        self._dec_inputs = []          # [(name, shape, type)]
        self._output_names = []
        self._embed_input = "input_ids"

    def is_loaded(self) -> bool:
        return self._loaded

    def _onnx_dir(self) -> str:
        d = Path(self.model_dir)
        sub = d / "onnx"
        return str(sub) if sub.is_dir() else str(d)

    def load(self) -> bool:
        """Laedt Embedding- + Decoder-Session + Tokenizer. Thread-safe, lazy."""
        with self._lock:
            if self._loaded:
                return True
            try:
                import onnxruntime as ort
                from tokenizers import Tokenizer
            except ImportError as e:
                logger.warning(f"⚠️ Korrektur-Dependencies fehlen ({e}) — Korrektur deaktiviert")
                return False

            try:
                t0 = time.time()
                onnx_dir = self._onnx_dir()
                emb_path = os.path.join(onnx_dir, f"embed_tokens_{self.quant}.onnx")
                dec_path = os.path.join(onnx_dir, f"decoder_model_merged_{self.quant}.onnx")
                tok_path = os.path.join(self.model_dir, "tokenizer.json")

                for p in (emb_path, dec_path, tok_path):
                    if not os.path.exists(p):
                        raise FileNotFoundError(p)

                so = ort.SessionOptions()
                so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                try:
                    so.intra_op_num_threads = max(1, (os.cpu_count() or 2) // 2)
                except Exception:
                    pass

                self._embed = ort.InferenceSession(emb_path, so, providers=["CPUExecutionProvider"])
                self._decoder = ort.InferenceSession(dec_path, so, providers=["CPUExecutionProvider"])
                self._tokenizer = Tokenizer.from_file(tok_path)
                self._embed_input = self._embed.get_inputs()[0].name
                self._dec_inputs = [(i.name, i.shape, i.type) for i in self._decoder.get_inputs()]
                self._output_names = [o.name for o in self._decoder.get_outputs()]
                self._loaded = True
                logger.info(
                    f"✅ TextCorrector geladen ({self.quant}) aus {self.model_dir} "
                    f"in {time.time() - t0:.1f}s"
                )
                return True
            except Exception as e:
                logger.warning(
                    f"⚠️ TextCorrector konnte nicht geladen werden: {e} — "
                    f"Korrektur deaktiviert (Fallback auf Rohtext)"
                )
                self._loaded = False
                return False

    def correct(self, text: str) -> str:
        """Korrigiert `text`. Bei jedem Fehler -> Originaltext (kein Crash)."""
        if not text or not text.strip():
            return text
        if not self._loaded and not self.load():
            return text
        try:
            return self._generate(text)
        except Exception as e:
            logger.warning(f"⚠️ Textkorrektur fehlgeschlagen: {e} — liefere Rohtext")
            return text

    # ------------------------------------------------------------------ #

    def _build_prompt(self, text: str) -> str:
        # /no_think schaltet den Reasoning-Modus von Qwen3.5 aus (schnell,
        # direkte Antwort). Als Sicherheitsnetz wird <think>...</think> spaeter
        # entfernt.
        user = text.strip() + " /no_think"
        return (
            f"<|im_start|>system\n{_SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    def _generate(self, text: str) -> str:
        prompt_ids = self._tokenizer.encode(self._build_prompt(text)).ids
        if not prompt_ids:
            return text

        # Embeddings des Prompts einmal berechnen; pro Schritt nur das neue
        # Token anhaengen (Embedding ist billig vs. Decoder, aber trotzdem
        # vermeiden wir O(n^2) im Embedding-Schritt).
        prompt_embeds = self._embed.run(
            None, {self._embed_input: np.array([prompt_ids], dtype=np.int64)}
        )[0].astype(np.float32)
        embeds = prompt_embeds

        # Token-Budget: Korrektur ≈ Eingabelaenge, mit Sicherheitsaufschlag,
        # gedeckelt (Full-Recompute wird ab ~300 Tokens spuerbar langsam).
        cap = min(384, max(128, int(len(prompt_ids) * 1.6)))
        generated: list[int] = []
        t0 = time.time()

        for _ in range(cap):
            feed = self._build_feed(embeds, embeds.shape[1])
            outs = self._decoder.run(None, feed)
            logits = outs[self._output_names.index("logits")]
            nxt = int(np.argmax(logits[0, -1]))

            if nxt in (EOS_TOKEN_ID, IM_END):
                break
            generated.append(nxt)

            new_emb = self._embed.run(
                None, {self._embed_input: np.array([[nxt]], dtype=np.int64)}
            )[0].astype(np.float32)
            embeds = np.concatenate([embeds, new_emb], axis=1)

        out = self._tokenizer.decode(generated) if generated else ""
        out = self._strip_thinking(out).strip()
        elapsed = time.time() - t0

        if not out:
            logger.info(f"✨ Korrektur leer ({elapsed:.1f}s) — behalte Rohtext")
            return text

        logger.info(
            f"✨ Korrektur ({len(generated)} tok, {elapsed:.1f}s): "
            f"'{text.strip()[:60]}' -> '{out[:60]}'"
        )
        return out

    def _build_feed(self, embeds: np.ndarray, seq_len: int) -> dict:
        """Decoder-Input-Dict: Embeddings + 3D-mRoPE-Positionen + geleerter Cache."""
        feed = {
            "inputs_embeds": embeds,
            "attention_mask": np.ones((1, seq_len), dtype=np.int64),
            # Qwen3.5 nutzt mRoPE mit 3 Achsen (T,H,W); Text-only -> alle gleich.
            "position_ids": np.broadcast_to(
                np.arange(seq_len, dtype=np.int64), (3, 1, seq_len)
            ).copy(),
        }
        for name, shape, typ in self._dec_inputs:
            if name in feed:
                continue
            dims = []
            for d in shape:
                if isinstance(d, int):
                    dims.append(d)
                elif d == "batch_size":
                    dims.append(1)
                elif d in ("sequence_length", "total_sequence_length"):
                    dims.append(seq_len)
                elif d == "past_sequence_length":
                    dims.append(0)   # Full-Recompute: kein KV-Cache aus Vergangenheit
                else:
                    dims.append(1)
            is_float = "float" in (typ or "")
            dt = np.float32 if is_float else np.int64
            feed[name] = np.zeros(dims, dtype=dt)
        return feed

    @staticmethod
    def _strip_thinking(text: str) -> str:
        # Vollstaendige <think>...</think> Bloecke entfernen ...
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        # ... und einen eventuell offenen <think>-Rest am Ende.
        text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
        return text


# Modul-Singleton (lazy, erst beim ersten correct()-Aufruf geladen)
text_corrector = TextCorrector()


def correct_text(text: str) -> str:
    """Komfort-Wrapper fuer die Transkriptions-Pipeline."""
    return text_corrector.correct(text)
