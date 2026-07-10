"""
Parakeet TDT STT Engine
Handles model loading and transcription using onnx-asr + Parakeet TDT v3

Modell-Speicherort: model/
  ├── encoder-model.int4.onnx       (373 MB, int4 Encoder)
  ├── decoder_joint-model.int8.onnx  (18 MB, int8 Decoder+Joint)
  ├── nemo128.int8.onnx              (41 KB, Mel-Preprocessor)
  ├── vocab.txt                      (92 KB, SentencePiece)
  └── config.json                    (97 B)
"""
import logging
import os
import numpy as np
from pathlib import Path
from typing import Optional, Dict

logger = logging.getLogger(__name__)

# NOTE: onnx_asr is imported LAZILY inside load_model(), not at module level.
# Importing it here would pull the full onnx_asr -> onnxruntime -> huggingface_hub
# graph into PyInstaller's static Analysis (which does an isolated import of each
# dependency). Loading onnxruntime's native DLLs during that isolated subprocess
# can stall/timeout the Analysis step on CI runners (Windows Defender scans the
# DLLs). Bundling is handled separately via the spec's hiddenimports + collect_all,
# so lazy import does not break the freeze.


_REQUIRED_PARAKEET_FILES = [
    "encoder-model.int4.onnx",
    "decoder_joint-model.int8.onnx",
    "vocab.txt",
]

# Registered onnx-asr model name for Parakeet TDT v3. onnx_asr.load_model takes
# the model name as the FIRST argument and the local directory as the SECOND
# (quantization as keyword). Passing the directory as the model name raises
# ModelNotSupportedError.
MODEL_ID = "nemo-parakeet-tdt-0.6b-v3"

_HF_FALLBACK_MODELS = [
    "nemo-parakeet-tdt-0.6b-v3",                  # registered name -> correctly-named istupakov export
    "istupakov/parakeet-tdt-0.6b-v3-onnx",
    "efederici/parakeet-tdt-0.6b-v3-onnx-int4",
]


class ParakeetEngine:
    """Parakeet TDT speech-to-text engine via onnx-asr.

    Implementiert dieselbe Schnittstelle wie WhisperEngine, sodass main.py
    minimal geändert werden muss.
    """

    def __init__(
        self,
        model_name: str = "default",
        model_size: str = None,
        device: str = "auto",
        compute_type: str = "auto",
    ):
        if model_size is not None:
            model_name = model_size
        self.model_name = model_name
        self.model_size = model_name
        self.device = device
        self.compute_type = compute_type
        self.model = None
        self.is_loaded = False
        self._original_device = device

    def load_model(self) -> bool:
        try:
            import onnx_asr
        except ImportError as e:
            logger.error(f"❌ onnx-asr is not installed! ({e})")
            return False

        if self.is_loaded:
            logger.info("Model already loaded")
            return True

        try:
            default_dir = self._get_default_models_dir()

            if default_dir and self._is_local_model_present(default_dir):
                logger.info(f"📥 Loading Parakeet from local directory: {default_dir}")
                self._ensure_compatible_local_files(default_dir)
                self.model = onnx_asr.load_model(MODEL_ID, str(default_dir), quantization="int4")
                self.is_loaded = True
                logger.info("✅ Parakeet model loaded (local, offline)")
                return True

            # HuggingFace-Download ist OPT-IN (Standard: manuell in model/).
            # Ohne PW_ALLOW_HF_DOWNLOAD=1 laedt die App NIEMALS automatisch und
            # schreibt daher auch nichts in den /models-Ordner.
            allow_hf = os.environ.get("PW_ALLOW_HF_DOWNLOAD", "").strip().lower() in (
                "1", "true", "yes", "on",
            )
            if allow_hf:
                for hf_name in _HF_FALLBACK_MODELS:
                    try:
                        logger.info(f"📥 Loading Parakeet from HuggingFace: {hf_name}")
                        self.model = onnx_asr.load_model(hf_name)
                        self.is_loaded = True
                        logger.info(f"✅ Parakeet model loaded (HF: {hf_name})")
                        return True
                    except Exception as e:
                        logger.warning(f"⚠️ {hf_name} failed: {e}, trying next...")

                logger.error("❌ All HuggingFace model loading strategies failed")
                return False

            logger.error(
                f"❌ Model not found in {default_dir}. Please place the Parakeet ONNX files "
                f"(encoder-model.int4.onnx, decoder_joint-model.int8.onnx, vocab.txt) manually "
                f"in that folder. Set PW_ALLOW_HF_DOWNLOAD=1 to allow automatic download."
            )
            return False

        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def _get_default_models_dir(self) -> Optional[Path]:
        try:
            from runtime_hooks.path_redirect import DEFAULT_MODELS_DIR
            return DEFAULT_MODELS_DIR
        except ImportError:
            return None

    def _ensure_compatible_local_files(self, model_dir: Path) -> None:
        """Make the local efederici int4/int8 export loadable by onnx-asr.

        onnx-asr's int4 path expects `decoder_joint-model.int4.onnx`, but the
        shipped efederici export names the decoder `decoder_joint-model.int8.onnx`.
        Copy it to the expected name (idempotent, non-destructive) so the model
        loads offline with no manual user step.
        """
        int8_dec = model_dir / "decoder_joint-model.int8.onnx"
        int4_dec = model_dir / "decoder_joint-model.int4.onnx"
        if int8_dec.exists() and not int4_dec.exists():
            import shutil
            logger.info(f"📋 Copying decoder for onnx-asr int4 naming: {int8_dec.name} -> {int4_dec.name}")
            shutil.copy2(int8_dec, int4_dec)

    def _is_local_model_present(self, model_dir: Path) -> bool:
        return all((model_dir / f).exists() for f in _REQUIRED_PARAKEET_FILES)

    def is_model_downloaded(self, model_name: str = None) -> bool:
        default_dir = self._get_default_models_dir()
        if default_dir:
            return self._is_local_model_present(default_dir)
        return False

    def transcribe_audio(
        self,
        audio_data: np.ndarray,
        language: Optional[str] = None,
        task: str = "transcribe"
    ) -> Dict:
        if not self.is_loaded:
            logger.warning("Model not loaded, loading now...")
            if not self.load_model():
                return {
                    "success": False,
                    "error": "Failed to load model",
                    "text": ""
                }

        try:
            logger.info(f"🎙️ Transcribing with Parakeet TDT...")
            logger.info(f"   Audio shape: {audio_data.shape}")
            logger.info(f"   Audio dtype: {audio_data.dtype}")

            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)

            if len(audio_data.shape) > 1:
                audio_data = audio_data.flatten()

            text = self.model.recognize(audio_data).strip()

            logger.info(f"✅ Transcription complete!")
            logger.info(f"   Text: {text[:100]}..." if len(text) > 100 else f"   Text: {text}")

            return {
                "success": True,
                "text": text,
                "segments": [],
                "language": "auto",
                "language_probability": 1.0,
                "duration": len(audio_data) / 16000,
            }

        except Exception as e:
            logger.error(f"❌ Transcription failed: {e}")
            import traceback
            logger.error(traceback.format_exc())

            return {
                "success": False,
                "error": str(e),
                "text": ""
            }

    def transcribe_chunk(self, audio_data: np.ndarray) -> Dict:
        """Für inkrementelle Transkription (falls später benötigt)."""
        min_samples = int(1.0 * 16000)
        if len(audio_data) < min_samples:
            logger.info(f"⏭️ Chunk zu kurz ({len(audio_data)/16000:.1f}s < 1.0s), übersprungen")
            return {
                "success": False,
                "skipped": True,
                "text": "",
                "error": "Chunk too short"
            }
        return self.transcribe_audio(audio_data)
