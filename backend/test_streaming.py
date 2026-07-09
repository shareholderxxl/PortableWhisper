"""
Tests für Pre-Load + Batch Transkription (Phase 1-2)

Testet:
1. AudioCapture.drain_available() — Queue-Drain Logik
2. ParakeetEngine.transcribe_chunk() — Mindestlängen-Prüfung
3. Pre-Load + Batch Logik — Modell-Laden, Audio-Fluss
"""
import sys
import asyncio
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

# ─── Module vor Import mocken ─────────────────────────────────────────────
_sounddevice_mock = MagicMock()
_onnx_asr_mock = MagicMock()
_onnxruntime_mock = MagicMock()

sys.modules.setdefault('sounddevice', _sounddevice_mock)
sys.modules.setdefault('onnx_asr', _onnx_asr_mock)
sys.modules.setdefault('onnxruntime', _onnxruntime_mock)

backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

_runtime_hooks_mock = MagicMock()
_runtime_hooks_mock.path_redirect = MagicMock()
_runtime_hooks_mock.path_redirect.get_logs_dir = MagicMock(return_value=Path('/tmp'))
_runtime_hooks_mock.path_redirect.MODELS_DIR = Path('/tmp/models')
_runtime_hooks_mock.path_redirect.DEFAULT_MODELS_DIR = Path('/tmp/models/default')
_runtime_hooks_mock.path_redirect.GPU_LIBS_DIR = Path('/tmp/gpu_libs')
sys.modules.setdefault('runtime_hooks', _runtime_hooks_mock)
sys.modules.setdefault('runtime_hooks.path_redirect', _runtime_hooks_mock.path_redirect)

gpu_manager_mock = MagicMock()
sys.modules.setdefault('gpu_manager', gpu_manager_mock)

from audio_capture import AudioCapture
from parakeet_engine import ParakeetEngine


# ═══════════════════════════════════════════════════════════════════════════
# TEST 1: AudioCapture.drain_available()
# ═══════════════════════════════════════════════════════════════════════════

class TestDrainAvailable:
    """Testet die non-blocking Queue-Drain Methode."""

    def test_empty_queue_returns_none(self):
        cap = AudioCapture()
        result = cap.drain_available()
        assert result is None

    def test_single_chunk(self):
        cap = AudioCapture()
        chunk = np.array([[0.1], [0.2], [0.3]], dtype=np.float32)
        cap.audio_queue.put(chunk)
        result = cap.drain_available()
        assert result is not None
        np.testing.assert_array_equal(result, chunk)
        assert cap.audio_queue.empty()

    def test_multiple_chunks_concatenated(self):
        cap = AudioCapture()
        chunk1 = np.array([[0.1], [0.2]], dtype=np.float32)
        chunk2 = np.array([[0.3], [0.4], [0.5]], dtype=np.float32)
        cap.audio_queue.put(chunk1)
        cap.audio_queue.put(chunk2)
        result = cap.drain_available()
        assert result is not None
        assert result.shape == (5, 1)

    def test_queue_emptied_after_drain(self):
        cap = AudioCapture()
        for i in range(5):
            cap.audio_queue.put(np.array([[float(i)]], dtype=np.float32))
        cap.drain_available()
        assert cap.audio_queue.empty()


# ═══════════════════════════════════════════════════════════════════════════
# TEST 2: ParakeetEngine.transcribe_chunk() — Mindestlängen-Prüfung
# ═══════════════════════════════════════════════════════════════════════════

class TestTranscribeChunkMinLength:
    """Testet die Mindestlängen-Prüfung von transcribe_chunk()."""

    def _make_engine_with_mock_model(self):
        engine = ParakeetEngine(model_size="default", device="cpu")
        engine.is_loaded = True
        engine.model = MagicMock()
        return engine

    def test_chunk_too_short_skipped(self):
        engine = self._make_engine_with_mock_model()
        short_audio = np.zeros(int(0.5 * 16000), dtype=np.float32)
        result = engine.transcribe_chunk(short_audio)
        assert result["success"] is False
        assert result.get("skipped") is True
        engine.model.transcribe.assert_not_called()

    def test_chunk_exactly_1_second_not_skipped(self):
        engine = self._make_engine_with_mock_model()
        audio_1s = np.zeros(16000, dtype=np.float32)
        mock_segment = MagicMock()
        mock_segment.text = " test "
        mock_info = MagicMock()
        mock_info.language = "de"
        mock_info.language_probability = 0.95
        engine.model.transcribe.return_value = ([mock_segment], mock_info)
        result = engine.transcribe_chunk(audio_1s)
        assert result["success"] is True
        assert result["text"] == "test"

    def test_chunk_not_loaded_returns_error(self):
        engine = ParakeetEngine(model_size="default", device="cpu")
        engine.is_loaded = False
        audio = np.zeros(16000, dtype=np.float32)
        result = engine.transcribe_chunk(audio)
        assert result["success"] is False
        assert "not loaded" in result.get("error", "").lower()

    def test_2d_audio_flattened(self):
        engine = self._make_engine_with_mock_model()
        audio_2d = np.zeros((16000, 1), dtype=np.float32)
        mock_segment = MagicMock()
        mock_segment.text = " test "
        mock_info = MagicMock()
        mock_info.language = "de"
        mock_info.language_probability = 0.9
        engine.model.transcribe.return_value = ([mock_segment], mock_info)
        result = engine.transcribe_chunk(audio_2d)
        assert result["success"] is True
        call_args = engine.model.transcribe.call_args[0]
        passed_audio = call_args[0]
        assert passed_audio.ndim == 1


# ═══════════════════════════════════════════════════════════════════════════
# TEST 3: Pre-Load + Batch Logik
# ═══════════════════════════════════════════════════════════════════════════

class TestPreLoadBatch:
    """Testet die Pre-Load + Batch Architektur."""

    def test_model_load_task_exists_as_global(self):
        """model_load_task Global sollte in main.py existieren."""
        import main as main_module
        assert hasattr(main_module, 'model_load_task')

    def test_no_streaming_globals(self):
        """Streaming-Globals sollten entfernt worden sein."""
        import main as main_module
        assert not hasattr(main_module, 'streaming_task')
        assert not hasattr(main_module, 'partial_transcript')
        assert not hasattr(main_module, 'streaming_failed')
        assert not hasattr(main_module, 'streaming_buffer')
        assert not hasattr(main_module, 'streaming_lock')

    def test_no_vad_constants(self):
        """VAD-Konstanten sollten entfernt worden sein."""
        import main as main_module
        assert not hasattr(main_module, 'VAD_RMS_THRESHOLD')
        assert not hasattr(main_module, 'VAD_SILENCE_DURATION')
        assert not hasattr(main_module, 'VAD_MIN_SPEECH_DURATION')
        assert not hasattr(main_module, 'VAD_POLL_INTERVAL')
        assert not hasattr(main_module, 'STREAMING_CHUNK_MAX_SECONDS')

    def test_load_model_async_exists(self):
        """_load_model_async() Funktion sollte existieren."""
        import main as main_module
        assert hasattr(main_module, '_load_model_async')
        assert callable(main_module._load_model_async)

    def test_no_streaming_worker(self):
        """_streaming_worker() sollte entfernt worden sein."""
        import main as main_module
        assert not hasattr(main_module, '_streaming_worker')


class TestStopCapturesAudio:
    """Testet dass /stop den Rückgabewert von stop_recording() nutzt (Bug 1 Fix)."""

    def test_stop_uses_stop_recording_return_value(self):
        """Der Quellcode von /stop sollte den Rückgabewert von stop_recording nutzen."""
        import main as main_module
        import inspect
        source = inspect.getsource(main_module.stop_recording)
        # Prüfe dass stop_recording's Rückgabewert einer Variable zugewiesen wird
        assert "audio_data = await loop.run_in_executor(None, audio_capture.stop_recording)" in source or \
               "audio_data = await loop.run_in_executor(None, audio_capture.stop_recording)" in source.replace(' ', ' ')


class TestStartChecksRecording:
    """Testet dass /start den Rückgabewert von start_recording() prüft (Bug 2 Fix)."""

    def test_start_checks_return_value(self):
        """Der Quellcode von /start sollte start_recording()'s Rückgabewert prüfen."""
        import main as main_module
        import inspect
        source = inspect.getsource(main_module.start_recording)
        assert "if not audio_capture.start_recording" in source


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
