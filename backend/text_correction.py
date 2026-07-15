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

Status & Modell-Verzeichnis:
  - Der Prompt (nur der inhaltliche System-Teil) ist editierbar und wird in
    der zentralen config.json persistiert. Das technische Geruest
    (<|im_start|>/im_end, /no_think) bleibt fix.
  - Das Modell-Verzeichnis wird robust aufgeloest (exakt qwen3.5-0.8b-onnx/
    oder Auto-Detect eines qwen3.5*-Ordners unter model/).
  - Ein Status-State-Machine (missing/loading/loaded/error) steuert die
    UI-Anzeige und das Startup-Pre-Load.

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

DEFAULT_QUANT = "q4"
DEFAULT_MODEL_SUBDIR = "qwen3.5-0.8b-onnx"
MAX_PROMPT_CHARS = 2000

DEFAULT_SYSTEM_PROMPT = (
    "Du bist ein Korrekturassistent fuer transkribierte deutsche Diktate. "
    "Korrigiere den Text: entferne Fuellwoerter und Disfluenzen (aehm, hm, ...), "
    "hebe Grammatik und Rechtschreibung, loese Selbstkorrekturen auf. Behalte "
    "Sinn, Stil und die urspruengliche Wortstellung bei. Gib NUR den "
    "korrigierten Text zurueck, ohne Erklaerung, ohne Anfuehrungszeichen."
)

# ---------------------------------------------------------------------------
# Editierbarer System-Prompt (nur Inhalt; technisches Geruest bleibt fix)
# ---------------------------------------------------------------------------
system_prompt: str = DEFAULT_SYSTEM_PROMPT


def set_system_prompt(prompt) -> str:
    """Setzt den System-Prompt (Inhalt). Leer -> Default, Cap MAX_PROMPT_CHARS."""
    global system_prompt
    p = (prompt or "").strip()
    if not p:
        p = DEFAULT_SYSTEM_PROMPT
    if len(p) > MAX_PROMPT_CHARS:
        p = p[:MAX_PROMPT_CHARS]
    system_prompt = p
    return system_prompt


def get_system_prompt() -> str:
    return system_prompt


def _resolve_model_dir() -> str:
    """Loest das Modell-Verzeichnis auf:
       1) model/qwen3.5-0.8b-onnx/ (mit tokenizer.json)
       2) Auto-Detect: beliebiger qwen3.5*-Ordner unter model/ mit tokenizer.json
       3) Default-Erwartung (wird als 'missing' gemeldet)."""
    exact = MODELS_DIR / DEFAULT_MODEL_SUBDIR
    if (exact / "tokenizer.json").exists():
        return str(exact)
    if MODELS_DIR.is_dir():
        try:
            for d in sorted(MODELS_DIR.iterdir()):
                if d.is_dir() and d.name.lower().startswith("qwen3.5") \
                        and (d / "tokenizer.json").exists():
                    return str(d)
        except OSError:
            pass
    return str(exact)  # Default-Erwartung (Status -> missing)


class TextCorrector:
    """Lazy-loaded Qwen3.5-0.8B Korrektur-Engine (plain onnxruntime, CPU)."""

    def __init__(self, model_dir=None, quant: str = DEFAULT_QUANT):
        self.quant = quant
        self.model_dir = model_dir          # None -> dynamisch aufgeloest
        self._lock = threading.Lock()
        self._loaded = False
        self._load_state = "missing"        # missing|loading|loaded|error
        self._load_error = ""
        self._embed = None
        self._decoder = None
        self._tokenizer = None
        self._dec_inputs = []               # [(name, shape, type)]
        self._output_names = []
        self._embed_input = "input_ids"

    # ---- Oeffentliche Status-API ---------------------------------------- #

    def is_loaded(self) -> bool:
        return self._loaded

    def get_model_dir(self) -> str:
        return self.model_dir if self.model_dir else _resolve_model_dir()

    def _resolve_layout(self):
        """Findet (onnx_dir, quant), indem Decoder+Embed gesucht werden.
        Kandidaten-Reihenfolge: onnx/ Subfolder (bevorzugte Quant, dann Fallback),
        danach flat im Modell-Root (gleiche Quant-Reihenfolge). So wird das Modell
        gefunden egal ob die Dateien in onnx/ ODER flach liegen, und egal ob
        q4 ODER q4f16. Bei Nichtfinden Default (onnx/<quant>) fuer die Meldung."""
        md = Path(self.get_model_dir())
        dirs = [md / "onnx", md]                      # onnx/ zuerst, dann flat
        quants = [self.quant, "q4f16"] if self.quant == "q4" else [self.quant, "q4"]
        for d in dirs:
            for q in quants:
                if (d / f"decoder_model_merged_{q}.onnx").exists() \
                        and (d / f"embed_tokens_{q}.onnx").exists():
                    return str(d), q
        return str(md / "onnx"), self.quant           # Default-Erwartung (-> 'missing')

    def _onnx_dir(self) -> str:
        return self._resolve_layout()[0]

    def get_layout(self):
        """-> (onnx_dir, quant): tatsaechlich aufgeloester Ablageort + Quant."""
        return self._resolve_layout()

    def check_files(self):
        """Prueft die benoetigten Modell-Dateien (auto-aufgeloest). -> (present, missing)."""
        od, q = self._resolve_layout()
        md = self.get_model_dir()
        required = [
            os.path.join(od, f"decoder_model_merged_{q}.onnx"),
            os.path.join(od, f"decoder_model_merged_{q}.onnx_data"),
            os.path.join(od, f"embed_tokens_{q}.onnx"),
            os.path.join(od, f"embed_tokens_{q}.onnx_data"),
            os.path.join(md, "tokenizer.json"),
        ]
        missing = [f for f in required if not os.path.exists(f)]
        return (len(missing) == 0), missing

    def get_status(self):
        """-> (state, missing_files). state: missing|available|loading|loaded|error."""
        if self._load_state == "loading":
            return "loading", []
        if self._loaded:
            return "loaded", []
        if self._load_state == "error":
            return "error", []
        present, missing = self.check_files()
        return ("available" if present else "missing"), missing

    # ---- Laden ---------------------------------------------------------- #

    def load(self) -> bool:
        """Laedt Embedding- + Decoder-Session + Tokenizer. Thread-safe, lazy."""
        with self._lock:
            if self._loaded:
                return True
            present, missing = self.check_files()
            if not present:
                self._load_state = "missing"
                logger.warning(f"⚠️ Korrektur-Modell unvollstaendig: {missing}")
                return False
            self._load_state = "loading"
            try:
                import onnxruntime as ort
                from tokenizers import Tokenizer
            except ImportError as e:
                self._loaded = False
                self._load_state = "error"
                self._load_error = str(e)
                logger.warning(f"⚠️ Korrektur-Dependencies fehlen ({e}) — deaktiviert")
                return False

            try:
                t0 = time.time()
                onnx_dir, q = self._resolve_layout()
                emb_path = os.path.join(onnx_dir, f"embed_tokens_{q}.onnx")
                dec_path = os.path.join(onnx_dir, f"decoder_model_merged_{q}.onnx")
                tok_path = os.path.join(self.get_model_dir(), "tokenizer.json")

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
                self.quant = q  # genutzte Variante merken (fuer Status/Meldung)
                self._loaded = True
                self._load_state = "loaded"
                self._load_error = ""
                logger.info(
                    f"✅ TextCorrector geladen ({q}) aus {self.get_model_dir()} "
                    f"in {time.time() - t0:.1f}s"
                )
                return True
            except Exception as e:
                self._loaded = False
                self._load_state = "error"
                self._load_error = str(e)
                logger.warning(
                    f"⚠️ TextCorrector konnte nicht geladen werden: {e} — "
                    f"Korrektur deaktiviert (Fallback auf Rohtext)"
                )
                return False

    # ---- Korrektur ------------------------------------------------------ #

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
        # entfernt. Das technische Geruest (im_start/im_end) bleibt fix.
        user = text.strip() + " /no_think"
        return (
            f"<|im_start|>system\n{get_system_prompt()}<|im_end|>\n"
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


# Modul-Singleton (lazy, Pre-Load beim App-Start oder beim ersten correct())
text_corrector = TextCorrector()


def correct_text(text: str) -> str:
    """Komfort-Wrapper fuer die Transkriptions-Pipeline."""
    return text_corrector.correct(text)
