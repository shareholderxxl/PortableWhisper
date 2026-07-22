"""
LLM Text Correction (Phase 3C — Qwen3 Migration).

Post-Processing des transkribierten Textes mit einem lokalen Qwen3-1.7B-
Modell (ONNX, q4f16) via plain onnxruntime — ohne onnxruntime-genai, ohne
genai-Format, ohne torch. Das Optimum-Export laeuft direkt auf der
onnxruntime CPU oder via DirectML auf der GPU (AMD/Intel/NVIDIA).

Architektur: Qwen3 ist ein Standard-Transformer (28 Layer full attention,
GQA). Das Single-File-ONNX-Format (`model_q4f16.onnx`) nimmt `input_ids`
direkt (Embedding integriert) — keine separate Embed-Session mehr.

  1. **KV-Cache + Prefix-Cache (Standard)** — _generate_kvcache():
     a) System-Prompt wird beim ersten Aufruf verarbeitet und als
        past_key_values zwischengespeichert (Prefix-Cache). Bei unverändertem
        Prompt wird er bei Folge-Korrekturen wiederverwendet.
     b) User-Prefill mit dem zwischengespeicherten System-Cache.
     c) Decode-Schleife mit nur 1 Token + wachsendem Cache.
     => O(n) pro Schritt statt O(n^2).

  2. **Full-Recompute (Fallback)** — _generate(): jeder Token-Schritt
     verarbeitet die komplette (wachsende) Sequenz mit geleertem Cache neu.
     O(n^2), aber einfacher und als Notfall-Fallback wichtig.

Execution Provider: DirectML (GPU) falls verfuegbar (onnxruntime-directml,
Default ON), sonst CPU. Die Auswahl passiert automatisch in
_select_providers() und kann per use_directml=False deaktiviert werden.

Status & Modell-Verzeichnis:
  - Der Prompt (nur der inhaltliche System-Teil) ist editierbar und wird in
    der zentralen config.json persistiert. Das technische Geruest
    (<|im_start|>/im_end, /no_think) bleibt fix.
  - Das Modell-Verzeichnis wird robust aufgeloest (exakt qwen3-1.7b-onnx/
    oder Auto-Detect eines qwen3-*-onnx-Ordners unter model/). Drop-In-
    kompatible Alternativen: Qwen3-0.6B-ONNX (570 MB), Qwen3-4B-ONNX (~2,8 GB).
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

# Qwen3 Special Tokens (siehe tokenizer_config.json / config.json).
# Bei Qwen3 gilt: IM_END == EOS_TOKEN_ID == 151645.
IM_START = 151644
IM_END = 151645
EOS_TOKEN_ID = 151645

DEFAULT_QUANT = "q4f16"
DEFAULT_MODEL_SUBDIR = "qwen3-1.7b-onnx"
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
    """Setzt den System-Prompt (Inhalt). Leer -> Default, Cap MAX_PROMPT_CHARS.
    Invalidiert den Prefix-Cache, damit der neue Prompt beim naechsten Aufruf
    neu verarbeitet wird."""
    global system_prompt
    p = (prompt or "").strip()
    if not p:
        p = DEFAULT_SYSTEM_PROMPT
    if len(p) > MAX_PROMPT_CHARS:
        p = p[:MAX_PROMPT_CHARS]
    system_prompt = p
    text_corrector._invalidate_prefix_cache()
    return system_prompt


def get_system_prompt() -> str:
    return system_prompt


# ---------------------------------------------------------------------------
# DirectML-Toggle (GPU an/aus). Default ON — Qwen3 (Standard Transformer)
# hat keine Precision-Probleme mit DirectML (im Gegensatz zu Qwen3.5-Hybrid).
# ---------------------------------------------------------------------------
use_directml: bool = True


def set_use_directml(value: bool) -> bool:
    global use_directml
    use_directml = bool(value)
    return use_directml


def get_use_directml() -> bool:
    return use_directml


def _resolve_model_dir() -> str:
    """Loest das Modell-Verzeichnis auf:
       1) model/qwen3-1.7b-onnx/ (Default, mit tokenizer.json)
       2) Auto-Detect: beliebiger qwen3-*-onnx Ordner unter model/ mit tokenizer.json
          (Drop-In-kompatibel: Qwen3-0.6B-ONNX, Qwen3-1.7B-ONNX, Qwen3-4B-ONNX)
       3) Default-Erwartung (wird als 'missing' gemeldet)."""
    exact = MODELS_DIR / DEFAULT_MODEL_SUBDIR
    if (exact / "tokenizer.json").exists():
        return str(exact)
    if MODELS_DIR.is_dir():
        try:
            for d in sorted(MODELS_DIR.iterdir()):
                if d.is_dir() and d.name.lower().startswith("qwen3-") \
                        and d.name.lower().endswith("-onnx") \
                        and (d / "tokenizer.json").exists():
                    return str(d)
        except OSError:
            pass
    return str(exact)  # Default-Erwartung (Status -> missing)


class TextCorrector:
    """Lazy-loaded Qwen3 Korrektur-Engine (onnxruntime, KV-Cache + Prefix-Cache
    + DirectML).

    Single-File-ONNX-Format: das Modell nimmt `input_ids` direkt entgegen
    (Embedding ist integriert) — keine separate Embed-Session mehr.
    """

    def __init__(self, model_dir=None, quant: str = DEFAULT_QUANT):
        self.quant = quant
        self.model_dir = model_dir          # None -> dynamisch aufgeloest
        self._lock = threading.Lock()
        self._loaded = False
        self._load_state = "missing"        # missing|loading|loaded|error
        self._load_error = ""
        self._decoder = None                # Single session (embed integriert)
        self._tokenizer = None
        self._dec_inputs = []               # [(name, shape, type)]
        self._output_names = []
        self._active_provider = ""          # fuer UI-Diagnose
        self._kvcache_was_empty = False
        # Prefix-Cache fuer System-Prompt (bleibt ueber mehrere correct()-Aufrufe)
        self._prefix_cache = None           # dict der past_key_values fuer System-Prefix
        self._prefix_len = 0                # Anzahl Token im System-Präfix
        self._prefix_hash = None            # Hash des System-Präfixes (Aenderungserkennung)

    # ---- Oeffentliche Status-API ---------------------------------------- #

    def is_loaded(self) -> bool:
        return self._loaded

    def get_model_dir(self) -> str:
        return self.model_dir if self.model_dir else _resolve_model_dir()

    def get_active_provider(self) -> str:
        """Aktiver Execution Provider fuer UI-Diagnose ('DmlExecutionProvider'
        oder 'CPUExecutionProvider')."""
        return self._active_provider

    def is_prefix_cache_active(self) -> bool:
        """True falls der Prefix-Cache fuer den System-Prompt aktiv ist."""
        return self._prefix_cache is not None

    def _resolve_layout(self):
        """Findet (onnx_dir, quant) fuer das Single-File-ONNX-Format.
        Sucht model_{quant}.onnx (ggf. mit externem _data Sidecar, der von
        onnxruntime automatisch geladen wird). Fallback-Kette:
        konfigurierte Quant -> q4f16 -> q4."""
        md = Path(self.get_model_dir())
        dirs = [md / "onnx", md]                      # onnx/ zuerst, dann flat
        quants = [self.quant] if self.quant == "q4f16" else [self.quant, "q4f16"]
        quants.append("q4")                            # letzter Fallback
        for d in dirs:
            for q in quants:
                if (d / f"model_{q}.onnx").exists():
                    return str(d), q
        return str(md / "onnx"), self.quant           # Default-Erwartung (-> 'missing')

    def _onnx_dir(self) -> str:
        return self._resolve_layout()[0]

    def get_layout(self):
        """-> (onnx_dir, quant): tatsaechlich aufgeloester Ablageort + Quant."""
        return self._resolve_layout()

    def check_files(self):
        """Prueft die benoetigten Modell-Dateien (Single-File-Format).
        Der optionale _data Sidecar (bei groesseren Modellen wie Qwen3-4B)
        wird von onnxruntime automatisch gefunden und muss hier nicht geprueft
        werden."""
        od, q = self._resolve_layout()
        md = self.get_model_dir()
        required = [
            os.path.join(od, f"model_{q}.onnx"),
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

    def _select_providers(self):
        """Waehlt den besten Execution Provider: DirectML (GPU) > CPU.
        Kann per use_directml-Flag deaktiviert werden (Diagnose).
        DirectML unterstuetzt jede DirectX-12-GPU (AMD, Intel, NVIDIA)."""
        if not use_directml:
            logger.info("💻 DirectML deaktiviert (config) — CPU-Modus")
            return ["CPUExecutionProvider"]
        try:
            import onnxruntime as ort
            available = ort.get_available_providers()
        except Exception:
            return ["CPUExecutionProvider"]
        if "DmlExecutionProvider" in available:
            logger.info("🖥️ DirectML verfügbar — nutze GPU (device_id=0)")
            return [("DmlExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
        logger.info("💻 DirectML nicht verfügbar — CPU-Modus")
        return ["CPUExecutionProvider"]

    def load(self) -> bool:
        """Laedt die Decoder-Session + Tokenizer. Thread-safe, lazy.
        Single-File-Format: das Embedding ist im Modell integriert — keine
        separate Embed-Session mehr (im Gegensatz zu Qwen3.5)."""
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
                model_path = os.path.join(onnx_dir, f"model_{q}.onnx")
                tok_path = os.path.join(self.get_model_dir(), "tokenizer.json")

                so = ort.SessionOptions()
                so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                try:
                    so.intra_op_num_threads = max(1, os.cpu_count() or 4)
                    so.inter_op_num_threads = 2
                    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                except Exception:
                    pass

                providers = self._select_providers()
                self._decoder = ort.InferenceSession(model_path, so, providers=providers)
                self._tokenizer = Tokenizer.from_file(tok_path)
                self._dec_inputs = [(i.name, i.shape, i.type) for i in self._decoder.get_inputs()]
                self._output_names = [o.name for o in self._decoder.get_outputs()]
                self.quant = q  # genutzte Variante merken (fuer Status/Meldung)
                self._loaded = True
                self._load_state = "loaded"
                self._load_error = ""
                # Prefix-Cache verwerfen (neues Modell)
                self._prefix_cache = None
                self._prefix_hash = None
                self._prefix_len = 0
                actual_ep = self._decoder.get_providers()
                self._active_provider = actual_ep[0] if actual_ep else "CPUExecutionProvider"
                ep_short = "GPU (DirectML)" if "Dml" in self._active_provider else "CPU"
                logger.info(
                    f"✅ TextCorrector geladen ({q}, EP={ep_short}) aus "
                    f"{self.get_model_dir()} in {time.time() - t0:.1f}s"
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
        """Korrigiert `text`. KV-Cache zuerst; bei Fehler oder leerem Output
        -> Full-Recompute-Fallback. Bei JEDEM Fehler -> Originaltext."""
        if not text or not text.strip():
            return text
        if not self._loaded and not self.load():
            return text
        self._kvcache_was_empty = False
        try:
            result = self._generate_kvcache(text)
            if not self._kvcache_was_empty:
                return result
            logger.info("🔄 KV-Cache lieferte leer — Full-Recompute-Fallback")
        except Exception as e:
            logger.warning(f"⚠️ KV-Cache-Korrektur fehlgeschlagen ({e}) — Full-Recompute-Fallback")
        try:
            return self._generate(text)
        except Exception as e2:
            logger.warning(f"⚠️ Textkorrektur fehlgeschlagen: {e2} — liefere Rohtext")
            return text

    # ------------------------------------------------------------------ #

    def _build_prompt(self, text: str) -> str:
        # /no_think schaltet den Reasoning-Modus von Qwen3 aus (schnell,
        # direkte Antwort). Als Sicherheitsnetz wird <think>...</think>
        # spaeter entfernt. Das technische Geruest (im_start/im_end) bleibt fix.
        user = text.strip() + " /no_think"
        return (
            f"<|im_start|>system\n{get_system_prompt()}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    def _build_prompt_parts(self, text: str):
        """Spaltet den Prompt in System-Präfix (gecacht) und User-Suffix (neu).
        Wird fuer den Prefix-Cache genutzt."""
        p = (system_prompt or DEFAULT_SYSTEM_PROMPT).strip()
        sys_prefix = (
            f"<|im_start|>system\n{p}<|im_end|>\n"
            f"<|im_start|>user\n"
        )
        user_suffix = f"{text.strip()} /no_think<|im_end|>\n<|im_start|>assistant\n"
        return sys_prefix, user_suffix

    # ---- KV-Cache Decode (Standard, schnell) --------------------------- #

    @staticmethod
    def _present_to_past(out_name: str):
        """Mappt present.{i}.key/value -> past_key_values.{i}.key/value.
        Returns None fuer 'logits' (kein Cache).
        Standard Transformer (Qwen3): nur present.* Outputs."""
        if out_name == "logits":
            return None
        if out_name.startswith("present."):
            return out_name.replace("present.", "past_key_values.", 1)
        return None

    def _zero_cache_inputs(self) -> dict:
        """Erzeugt geleerte past_key_values fuer den Prefill-Pass
        (past_sequence_length=0 fuer alle KV-Cache-Layer)."""
        cache = {}
        for name, shape, typ in self._dec_inputs:
            if name in ("input_ids", "attention_mask", "position_ids"):
                continue
            # Alle anderen Inputs sind past_key_values.{i}.key|value
            dims = []
            for d in shape:
                if isinstance(d, int):
                    dims.append(d)
                elif d == "batch_size":
                    dims.append(1)
                elif d == "past_sequence_length":
                    dims.append(0)        # leerer Prefix
                else:
                    dims.append(1)
            # q4f16-Modelle haben float16-Aktivierungen/KV-Cache; q4 hat float32.
            # Der ONNX-Typ-String ist z.B. "tensor(float16)" oder "tensor(float)".
            if "float16" in (typ or ""):
                dt = np.float16
            elif "float" in (typ or ""):
                dt = np.float32
            else:
                dt = np.int64
            cache[name] = np.zeros(dims, dtype=dt)
        return cache

    def _get_or_build_prefix_cache(self, sys_ids):
        """Gibt den gecachten System-Prompt-Cache zurueck, oder baut ihn neu auf.
        System-Prompt aendert sich selten — Cache-Hit spart ~0,7-1,5s Prefill."""
        prompt_hash = hash(tuple(sys_ids))
        if self._prefix_cache is not None and self._prefix_hash == prompt_hash:
            logger.info(f"🔧 Prefix-Cache hit ({self._prefix_len} tok übersprungen)")
            return self._prefix_cache, self._prefix_len

        # Cache neu aufbauen (nur System-Praefix prefillen)
        t0 = time.time()
        N_sys = len(sys_ids)
        feed = {
            "input_ids": np.array([sys_ids], dtype=np.int64),
            "attention_mask": np.ones((1, N_sys), dtype=np.int64),
            "position_ids": np.arange(N_sys, dtype=np.int64).reshape(1, N_sys),
        }
        feed.update(self._zero_cache_inputs())
        outs = self._decoder.run(None, feed)

        cache = {}
        for name, val in zip(self._output_names, outs):
            past = self._present_to_past(name)
            if past:
                cache[past] = val  # ORT pure — Cache bleibt intakt

        self._prefix_cache = cache
        self._prefix_len = N_sys
        self._prefix_hash = prompt_hash
        logger.info(f"🔧 Prefix-Cache aufgebaut ({N_sys} tok in {time.time()-t0:.2f}s)")
        return cache, N_sys

    def _invalidate_prefix_cache(self):
        """Verwirft den Prefix-Cache. Wird bei System-Prompt-Aenderung oder
        Modell-Reload aufgerufen."""
        if self._prefix_cache is not None:
            logger.info("🔧 Prefix-Cache invalidiert")
        self._prefix_cache = None
        self._prefix_hash = None
        self._prefix_len = 0

    def _generate_kvcache(self, text: str) -> str:
        """KV-Cache inkrementelles Decoding mit Prefix-Cache fuer System-Prompt.

        1. System-Prompt-Token aus dem Prefix-Cache holen (oder neu aufbauen).
        2. User-Suffix mit dem Prefix-Cache prefillen.
        3. Decode-Loop mit nur 1 Token + wachsendem Cache.
        """
        # Prompt in System-Praefix (gecacht) und User-Suffix (neu) spalten
        sys_prefix_str, user_suffix_str = self._build_prompt_parts(text)
        sys_ids = self._tokenizer.encode(sys_prefix_str).ids
        user_ids = self._tokenizer.encode(user_suffix_str).ids

        # Tokenisierungs-Grenze validieren: selten, aber manche Tokenizer
        # mergen an der Grenze. Falls gemerged -> Full-Prefill-Fallback.
        full_ids = self._tokenizer.encode(self._build_prompt(text)).ids
        if sys_ids + user_ids != full_ids:
            logger.info("⚠️ Tokenizer-Grenze verschoben — Full-Prefill (kein Prefix-Cache)")
            return self._generate_kvcache_full_prefill(text)

        # Prefix-Cache holen (oder neu aufbauen)
        prefix_cache, N_sys = self._get_or_build_prefix_cache(sys_ids)

        N_user = len(user_ids)
        N_total = N_sys + N_user
        cap = min(256, max(64, int(N_total * 1.5)))
        generated: list[int] = []
        eos_hit = False
        t0 = time.time()

        # --- USER-PREFILL: mit System-Cache, nur User-Token neu ---
        feed = {
            "input_ids": np.array([user_ids], dtype=np.int64),
            "attention_mask": np.ones((1, N_total), dtype=np.int64),
            "position_ids": np.arange(N_sys, N_sys + N_user, dtype=np.int64).reshape(1, N_user),
        }
        feed.update(prefix_cache)   # System-Prompt Cache injizieren

        outs = self._decoder.run(None, feed)
        prefill_dt = time.time() - t0
        logits = outs[self._output_names.index("logits")]
        nxt = int(np.argmax(logits[0, -1]))

        # Cache aus Outputs extrahieren (System + User, wachsend)
        cache = {}
        for name, val in zip(self._output_names, outs):
            past = self._present_to_past(name)
            if past:
                cache[past] = val

        if nxt in (EOS_TOKEN_ID, IM_END):
            eos_hit = True
            logger.info(f"⏱️ KV-Cache prefill: {N_user}+{N_sys} tok in {prefill_dt:.2f}s (sofort EOS, Prefix-Cache hit)")
            return self._finalize(text, generated, t0, "KV-Cache", eos_hit)
        generated.append(nxt)

        # --- DECODE-LOOP: 1 Token + Cache ---
        decode_t0 = time.time()
        total_len = N_total
        for _ in range(cap - 1):
            total_len += 1
            feed = {
                "input_ids": np.array([[nxt]], dtype=np.int64),
                "attention_mask": np.ones((1, total_len), dtype=np.int64),
                "position_ids": np.array([[total_len - 1]], dtype=np.int64),
            }
            feed.update(cache)

            outs = self._decoder.run(None, feed)
            logits = outs[self._output_names.index("logits")]
            nxt = int(np.argmax(logits[0, 0]))

            for name, val in zip(self._output_names, outs):
                past = self._present_to_past(name)
                if past:
                    cache[past] = val

            if nxt in (EOS_TOKEN_ID, IM_END):
                eos_hit = True
                break
            generated.append(nxt)

        decode_dt = time.time() - decode_t0
        logger.info(
            f"⏱️ KV-Cache: prefill {N_user}+{N_sys} tok in {prefill_dt:.2f}s, "
            f"decode {len(generated)} tok in {decode_dt:.2f}s "
            f"({len(generated)/max(decode_dt,0.001):.1f} tok/s, EOS={eos_hit})"
        )
        return self._finalize(text, generated, t0, "KV-Cache", eos_hit)

    def _generate_kvcache_full_prefill(self, text: str) -> str:
        """KV-Cache Decode OHNE Prefix-Cache (Fallback, falls Tokenizer-Grenze
        verschoben ist). Verarbeitet den kompletten Prompt neu."""
        prompt_ids = self._tokenizer.encode(self._build_prompt(text)).ids
        if not prompt_ids:
            return text

        N = len(prompt_ids)
        cap = min(256, max(64, int(N * 1.5)))
        generated: list[int] = []
        eos_hit = False
        t0 = time.time()

        # --- PREFILL: voller Prompt, geleerte Caches ---
        feed = {
            "input_ids": np.array([prompt_ids], dtype=np.int64),
            "attention_mask": np.ones((1, N), dtype=np.int64),
            "position_ids": np.arange(N, dtype=np.int64).reshape(1, N),
        }
        feed.update(self._zero_cache_inputs())

        outs = self._decoder.run(None, feed)
        prefill_dt = time.time() - t0
        logits = outs[self._output_names.index("logits")]
        nxt = int(np.argmax(logits[0, -1]))

        cache = {}
        for name, val in zip(self._output_names, outs):
            past = self._present_to_past(name)
            if past:
                cache[past] = val

        if nxt in (EOS_TOKEN_ID, IM_END):
            eos_hit = True
            logger.info(f"⏱️ KV-Cache full-prefill: {N} tok in {prefill_dt:.2f}s (sofort EOS)")
            return self._finalize(text, generated, t0, "KV-Cache", eos_hit)
        generated.append(nxt)

        # --- DECODE-LOOP: 1 Token + Cache ---
        decode_t0 = time.time()
        total_len = N
        for _ in range(cap - 1):
            total_len += 1
            feed = {
                "input_ids": np.array([[nxt]], dtype=np.int64),
                "attention_mask": np.ones((1, total_len), dtype=np.int64),
                "position_ids": np.array([[total_len - 1]], dtype=np.int64),
            }
            feed.update(cache)

            outs = self._decoder.run(None, feed)
            logits = outs[self._output_names.index("logits")]
            nxt = int(np.argmax(logits[0, 0]))

            for name, val in zip(self._output_names, outs):
                past = self._present_to_past(name)
                if past:
                    cache[past] = val

            if nxt in (EOS_TOKEN_ID, IM_END):
                eos_hit = True
                break
            generated.append(nxt)

        decode_dt = time.time() - decode_t0
        logger.info(
            f"⏱️ KV-Cache (full-prefill): prefill {N} tok in {prefill_dt:.2f}s, "
            f"decode {len(generated)} tok in {decode_dt:.2f}s "
            f"({len(generated)/max(decode_dt,0.001):.1f} tok/s, EOS={eos_hit})"
        )
        return self._finalize(text, generated, t0, "KV-Cache", eos_hit)

    def _finalize(self, text: str, generated: list[int], t0: float,
                  method: str = "", eos_hit: bool = False) -> str:
        """Decode + Log (shared von _generate_kvcache und _generate)."""
        raw = self._tokenizer.decode(generated) if generated else ""
        out = self._strip_thinking(raw).strip()
        elapsed = time.time() - t0
        if not out:
            logger.info(
                f"✨ {method} leer ({elapsed:.1f}s, {len(generated)} tok, EOS={eos_hit})"
            )
            if method == "KV-Cache":
                logger.info(f"   Raw output: {raw[:150]!r}")
                logger.info(f"   First tokens: {generated[:15]}")
                self._kvcache_was_empty = True
            return text
        logger.info(
            f"✨ {method} ({len(generated)} tok, {elapsed:.1f}s): "
            f"'{text.strip()[:60]}' -> '{out[:60]}'"
        )
        return out

    # ---- Full-Recompute Decode (Fallback) ------------------------------ #

    def _generate(self, text: str) -> str:
        """Full-Recompute-Fallback: jeder Schritt verarbeitet die komplette
        Sequenz neu (O(n^2)). Wird nur genutzt, falls _generate_kvcache()
        fehlschlaegt oder leer liefert."""
        prompt_ids = self._tokenizer.encode(self._build_prompt(text)).ids
        if not prompt_ids:
            return text

        all_ids = list(prompt_ids)
        cap = min(384, max(128, int(len(prompt_ids) * 1.6)))
        generated: list[int] = []
        eos_hit = False
        t0 = time.time()

        for _ in range(cap):
            feed = self._build_feed(np.array([all_ids], dtype=np.int64), len(all_ids))
            outs = self._decoder.run(None, feed)
            logits = outs[self._output_names.index("logits")]
            nxt = int(np.argmax(logits[0, -1]))

            if nxt in (EOS_TOKEN_ID, IM_END):
                eos_hit = True
                break
            generated.append(nxt)
            all_ids.append(nxt)

        return self._finalize(text, generated, t0, "Full-Recompute", eos_hit)

    def _build_feed(self, input_ids: np.ndarray, seq_len: int) -> dict:
        """Decoder-Input-Dict: input_ids + 2D position_ids + geleerter Cache.
        Full-Recompute-Modus: kein KV-Cache aus der Vergangenheit."""
        feed = {
            "input_ids": input_ids,
            "attention_mask": np.ones((1, seq_len), dtype=np.int64),
            "position_ids": np.arange(seq_len, dtype=np.int64).reshape(1, seq_len),
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
            # Gleiche float16/float32-Logik wie _zero_cache_inputs()
            if "float16" in (typ or ""):
                dt = np.float16
            elif "float" in (typ or ""):
                dt = np.float32
            else:
                dt = np.int64
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
