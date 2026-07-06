"""
PortableWhisper Backend Server
FastAPI server for local speech-to-text processing
"""

# Pfad-Management GANZ OBEN importieren: setzt %LOCALAPPDATA%-Pfade und
# HuggingFace-Cache-Umgebungsvariablen, BEVOR faster_whisper/huggingface_hub
# geladen werden. Muss vor jedem anderen App-Modul kommen.
from runtime_hooks import path_redirect  # noqa: F401  (Seiteneffekt-Import)

import logging
import asyncio
import time
from contextlib import asynccontextmanager
from typing import Optional, Dict, List
import numpy as np

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import our modules
from audio_capture import AudioCapture
from whisper_engine import WhisperEngine
import gpu_manager

# Konfiguration
BACKEND_HOST = "127.0.0.1"  # localhost only — kein externer Zugriff
BACKEND_PORT = 8765         # Sidecar-Port (Phase 1)

import sys
from runtime_hooks.path_redirect import get_logs_dir

# Configure logging
log_file = get_logs_dir() / "whisper-backend.log"
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler(log_file, encoding="utf-8", delay=True)
file_handler.setFormatter(file_formatter)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        file_handler
    ]
)
logger = logging.getLogger(__name__)

# Route external loggers to our file handler
for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "faster_whisper", "huggingface_hub"):
    logging.getLogger(name).addHandler(file_handler)

# Global instances
audio_capture: Optional[AudioCapture] = None
whisper_engine: Optional[WhisperEngine] = None
is_recording = False
transcription_task: Optional[asyncio.Task] = None
last_transcribed_text = ""
is_model_loading = False
model_loading_info = {"model": "", "status": ""}
current_language: Optional[str] = None  # Store language from start request

# Streaming state
streaming_task: Optional[asyncio.Task] = None
model_load_task: Optional[asyncio.Task] = None
partial_transcript: str = ""
streaming_lock: Optional[asyncio.Lock] = None
streaming_failed: bool = False
streaming_buffer: List[np.ndarray] = []

# VAD constants
STREAMING_CHUNK_MAX_SECONDS = 25
VAD_RMS_THRESHOLD = 0.02
VAD_SILENCE_DURATION = 0.4
VAD_MIN_SPEECH_DURATION = 1.0
VAD_POLL_INTERVAL = 0.1


# Pydantic models
class StartRequest(BaseModel):
    model_size: str = "default"  # "default" = data/models/default/ Ordner
    language: Optional[str] = None  # Language code or None for auto-detect
    device: str = "auto"  # auto, cpu, cuda
    device_index: Optional[int] = None  # Microphone device index (None = default)


class StopRequest(BaseModel):
    pass


class TranscriptionResponse(BaseModel):
    success: bool
    text: str = ""
    is_final: bool = False
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    backend: str
    model: str
    model_status: str = "missing"
    recording: bool = False
    version: str = "1.0.0"
    recording: bool


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for the FastAPI app"""
    logger.info("=" * 60)
    logger.info("🚀 PortableWhisper Backend Starting...")
    logger.info("=" * 60)
    logger.info(f"Server: http://{BACKEND_HOST}:{BACKEND_PORT}")
    logger.info(f"API Docs: http://{BACKEND_HOST}:{BACKEND_PORT}/docs")
    logger.info(f"Health Check: http://{BACKEND_HOST}:{BACKEND_PORT}/health")
    logger.info("=" * 60)

    # Check GPU libraries at startup
    logger.info("🔍 Checking GPU libraries...")
    gpu_info = gpu_manager.get_gpu_info()

    if gpu_info["gpu_available"]:
        logger.info("✅ NVIDIA GPU detected")
        if gpu_info["libs_installed"]:
            logger.info("✅ GPU libraries installed - GPU acceleration available")
        else:
            logger.warning("⚠️ GPU detected but libraries not installed")
            if gpu_info["missing_libraries"]:
                logger.warning(f"   Missing: {', '.join(gpu_info['missing_libraries'])}")
            logger.warning("   Use 'Install GPU Libraries' in settings to enable GPU acceleration")
    else:
        logger.info("ℹ️ No NVIDIA GPU detected - will use CPU mode")

    logger.info("=" * 60)

    yield

    # Cleanup on shutdown
    logger.info("Shutting down...")
    global audio_capture
    if is_recording and audio_capture:
        try:
            audio_capture.stop_recording()
        except:
            pass


# Create FastAPI app
app = FastAPI(
    title="PortableWhisper Backend",
    description="Local speech-to-text processing server",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for local development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "app": "PortableWhisper Backend",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint with model and device status.

    Wird vom Tauri-Frontend periodisch gepollt. model_status ist:
      - 'loaded'    : Modell im Speicher aktiv
      - 'available' : Modell lokal vorhanden, aber nicht geladen
      - 'missing'   : Modell fehlt in data/models/default/
    """
    global whisper_engine

    backend = "cpu"
    model = "default"
    model_status = "missing"

    if whisper_engine is not None:
        backend = str(whisper_engine.device)
        if whisper_engine.is_loaded:
            model = str(whisper_engine.model_size)
            model_status = "loaded"
        elif whisper_engine.is_model_downloaded():
            model_status = "available"

    return HealthResponse(
        status="ok",
        backend=backend,
        model=model,
        model_status=model_status,
        recording=is_recording
    )


@app.get("/model/status")
async def get_model_status(model_size: str = "small"):
    """Check if a model is downloaded and get loading status"""
    global is_model_loading, model_loading_info, whisper_engine

    try:
        # Check if model is currently loading
        if is_model_loading:
            return {
                "success": True,
                "is_loading": True,
                "loading_model": model_loading_info.get("model", ""),
                "status": model_loading_info.get("status", "Downloading...")
            }

        # Check if this specific model is downloaded
        temp_engine = WhisperEngine(model_size=model_size)
        is_downloaded = temp_engine.is_model_downloaded(model_size)

        # Check if model is currently loaded in memory
        is_loaded = whisper_engine is not None and whisper_engine.model_size == model_size and whisper_engine.is_loaded

        return {
            "success": True,
            "is_loading": False,
            "is_downloaded": is_downloaded,
            "is_loaded": is_loaded,
            "model": model_size
        }

    except Exception as e:
        logger.error(f"Error checking model status: {e}")
        return {
            "success": False,
            "error": str(e),
            "is_loading": False,
            "is_downloaded": False
        }


@app.post("/load_model")
async def load_model(request: StartRequest):
    """Pre-load a Whisper model without starting recording"""
    global whisper_engine, is_model_loading, model_loading_info, current_language

    try:
        logger.info(f"📥 Loading model: {request.model_size} on {request.device}")

        # Store language
        current_language = None if request.language in (None, "auto") else request.language

        # Check if we can reuse existing engine
        if whisper_engine is not None and \
           whisper_engine.model_size == request.model_size and \
           whisper_engine._original_device == request.device and \
           whisper_engine.is_loaded:
            logger.info(f"♻️ Model already loaded: {request.model_size} on {whisper_engine.device}")
            return {
                "status": "success",
                "message": "Model already loaded",
                "model": request.model_size,
                "device": whisper_engine.device
            }

        # Create new engine if needed
        if whisper_engine is None or \
           whisper_engine.model_size != request.model_size or \
           whisper_engine._original_device != request.device:
            whisper_engine = WhisperEngine(
                model_size=request.model_size,
                device=request.device
            )
            logger.info(f"✓ Whisper engine created (device: {whisper_engine.device})")

        # Load the model if not already loaded
        if not whisper_engine.is_loaded:
            is_model_loading = True
            model_loading_info = {
                "model": whisper_engine.model_size,
                "status": "Loading model..."
            }

            logger.info("📥 Loading Whisper model into memory...")
            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(None, whisper_engine.load_model)

            is_model_loading = False
            model_loading_info = {"model": "", "status": ""}

            if not success:
                error_msg = "Failed to load Whisper model."
                if whisper_engine.model_size.lower() == "default" and not whisper_engine.is_model_downloaded():
                    error_msg += " Please place CTranslate2 model files in data/models/default/"
                return {
                    "status": "error",
                    "message": error_msg,
                    "details": "Auto-download is disabled. Place models in data/models/default/."
                }

            logger.info(f"✅ Model loaded: {request.model_size} on {whisper_engine.device}")

        return {
            "status": "success",
            "message": "Model loaded successfully",
            "model": request.model_size,
            "device": whisper_engine.device
        }

    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        import traceback
        logger.error(traceback.format_exc())

        is_model_loading = False
        model_loading_info = {"model": "", "status": ""}

        return {
            "status": "error",
            "message": str(e)
        }


@app.post("/start")
async def start_recording(request: StartRequest):
    """Start recording with VAD-based streaming transcription"""
    global audio_capture, whisper_engine, is_recording, current_language
    global streaming_task, model_load_task, partial_transcript, streaming_lock
    global streaming_failed, streaming_buffer

    try:
        if is_recording:
            return {"status": "error", "message": "Already recording"}

        # Store language for use in worker + /stop
        current_language = None if request.language in (None, "auto") else request.language
        logger.info(f"🎙️ Starting recording (streaming VAD transcription)")
        logger.info(f"📋 Requested device: {request.device}")
        logger.info(f"🌐 Language: {current_language or 'auto-detect'}")

        # Reuse existing engine if model/device match, otherwise create new one
        if whisper_engine is not None and \
           whisper_engine.model_size == request.model_size and \
           whisper_engine._original_device == request.device:
            logger.info(f"♻️ Reusing existing Whisper engine (device: {whisper_engine.device})")
        else:
            whisper_engine = WhisperEngine(
                model_size=request.model_size,
                device=request.device
            )
            logger.info(f"✓ Whisper engine created (device: {whisper_engine.device})")

        # Initialize audio capture
        audio_capture = AudioCapture()
        audio_capture.clear_queue()

        device_index = request.device_index if request.device_index is not None else None
        if device_index is not None:
            logger.info(f"🎤 Using microphone device index: {device_index}")
        else:
            logger.info(f"🎤 Using default microphone device")

        audio_capture.start_recording(device_index=device_index)
        await asyncio.sleep(0.1)

        is_recording = True

        # Reset streaming state
        partial_transcript = ""
        streaming_failed = False
        streaming_buffer = []
        streaming_lock = asyncio.Lock()

        # Start model loading in background (lazy parallel)
        loop = asyncio.get_event_loop()
        model_load_task = asyncio.create_task(
            _load_model_async(loop)
        )

        # Start streaming worker
        streaming_task = asyncio.create_task(
            _streaming_worker(loop)
        )

        logger.info("✅ Recording started (model loading in background, streaming active)")

        return {
            "status": "started",
            "message": "Recording... Press Alt+T when done",
            "model": request.model_size,
            "device": whisper_engine.device
        }

    except Exception as e:
        logger.error(f"❌ Failed to start recording: {e}")
        import traceback
        logger.error(traceback.format_exc())

        is_recording = False
        if audio_capture:
            try:
                audio_capture.stop_recording()
            except:
                pass

        return {"status": "error", "message": str(e)}


async def _load_model_async(loop: asyncio.AbstractEventLoop):
    """Lädt das Whisper-Modell asynchron im Hintergrund."""
    global whisper_engine, is_model_loading, model_loading_info, streaming_failed

    try:
        is_model_loading = True
        model_loading_info = {
            "model": whisper_engine.model_size if whisper_engine else "default",
            "status": "Loading model..."
        }
        logger.info("📥 Loading Whisper model (background async)...")
        success = await loop.run_in_executor(None, whisper_engine.load_model)
        if success:
            logger.info("✅ Model loaded successfully (background)")
        else:
            logger.error("❌ Model loading failed (background)")
            streaming_failed = True
    except Exception as e:
        logger.error(f"❌ Model loading error: {e}")
        streaming_failed = True
    finally:
        is_model_loading = False
        model_loading_info = {"model": "", "status": ""}


async def _streaming_worker(loop: asyncio.AbstractEventLoop):
    """VAD-basierter Streaming-Worker. Läuft während is_recording == True."""
    global is_recording, streaming_buffer, partial_transcript, streaming_failed

    accumulated = []
    in_speech = False
    last_speech_time = time.time()

    try:
        while is_recording and not streaming_failed:
            await asyncio.sleep(VAD_POLL_INTERVAL)

            # Modell noch nicht bereit → Audio nur sammeln
            if not whisper_engine or not whisper_engine.is_loaded:
                chunk = audio_capture.drain_available()
                if chunk is not None:
                    accumulated.append(chunk)
                continue

            chunk = audio_capture.drain_available()
            if chunk is not None:
                accumulated.append(chunk)

            if not accumulated:
                continue

            full_audio = np.concatenate(accumulated, axis=0)
            total_duration = len(full_audio) / 16000

            recent_samples = min(int(0.2 * 16000), len(full_audio))
            recent_audio = full_audio[-recent_samples:]
            rms = float(np.sqrt(np.mean(recent_audio ** 2)))

            now = time.time()

            if rms > VAD_RMS_THRESHOLD:
                in_speech = True
                last_speech_time = now
            elif in_speech:
                silence_duration = now - last_speech_time
                if silence_duration >= VAD_SILENCE_DURATION:
                    speech_duration = len(full_audio) / 16000
                    if speech_duration >= VAD_MIN_SPEECH_DURATION:
                        try:
                            async with streaming_lock:
                                result = await loop.run_in_executor(
                                    None,
                                    whisper_engine.transcribe_chunk,
                                    full_audio.copy(),
                                    current_language if current_language != "en" else None,
                                    "translate" if current_language == "en" else "transcribe"
                                )
                            if result.get("success") and result.get("text"):
                                partial_transcript += result["text"] + " "
                        except Exception as e:
                            logger.error(f"Streaming transcribe error: {e}")
                            streaming_failed = True
                            break

                    accumulated = []
                    in_speech = False
                    last_speech_time = now
                    continue  # Verhindert Hard-Cap mit veralteten Werten

            # Hard-Cap: Whisper-Limit von 30s → bei 25s transkribieren
            if total_duration >= STREAMING_CHUNK_MAX_SECONDS:
                try:
                    async with streaming_lock:
                        result = await loop.run_in_executor(
                            None,
                            whisper_engine.transcribe_chunk,
                            full_audio.copy(),
                            current_language if current_language != "en" else None,
                            "translate" if current_language == "en" else "transcribe"
                        )
                    if result.get("success") and result.get("text"):
                        partial_transcript += result["text"] + " "
                except Exception as e:
                    logger.error(f"Streaming forced chunk error: {e}")
                    streaming_failed = True
                    break

                accumulated = []
                in_speech = False
                last_speech_time = now
    finally:
        # Stellt sicher, dass remaining Audio auch bei Cancel verfügbar ist
        if accumulated:
            streaming_buffer = accumulated


@app.post("/stop")
async def stop_recording():
    """Stop recording: streaming path if chunks were transcribed, else batch fallback"""
    global is_recording, audio_capture, whisper_engine
    global streaming_task, model_load_task, partial_transcript, streaming_failed
    global streaming_buffer

    try:
        if not is_recording:
            return {"status": "error", "message": "Not recording"}

        logger.info("🛑 Stopping recording...")
        is_recording = False

        # Wait for streaming worker to finish (timeout 10s for in-flight transcribe)
        if streaming_task:
            streaming_task.cancel()
            try:
                await asyncio.wait_for(streaming_task, timeout=10.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            streaming_task = None

        # Wait for model loading if still in progress
        if model_load_task and not (whisper_engine and whisper_engine.is_loaded):
            model_load_task.cancel()
            try:
                await asyncio.wait_for(model_load_task, timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            model_load_task = None

        # Stop audio capture
        loop = asyncio.get_event_loop()
        if audio_capture:
            await loop.run_in_executor(None, audio_capture.stop_recording)

        # Drain any remaining audio from queue + streaming_buffer
        remaining_chunks = list(streaming_buffer)
        final_chunk = audio_capture.drain_available() if audio_capture else None
        if final_chunk is not None:
            remaining_chunks.append(final_chunk)

        # --- DECISION: streaming path or batch fallback? ---
        use_streaming = not streaming_failed and bool(partial_transcript.strip())

        if use_streaming:
            logger.info("📝 Using streaming path — transcribing remaining buffer...")
            final_text = partial_transcript.strip()

            if remaining_chunks:
                rest_audio = np.concatenate(remaining_chunks, axis=0)
                rest_duration = len(rest_audio) / 16000
                if rest_duration >= VAD_MIN_SPEECH_DURATION:
                    # Ensure model is loaded
                    if not whisper_engine.is_loaded:
                        logger.info("📥 Loading model for rest flush...")
                        success = await loop.run_in_executor(None, whisper_engine.load_model)
                        if not success:
                            logger.warning("⚠️ Model load failed, returning partial transcript")
                            return {
                                "status": "success",
                                "text": final_text,
                                "partial": True,
                                "note": "Partial result (model load failed)"
                            }

                    async with streaming_lock:
                        rest_result = await loop.run_in_executor(
                            None,
                            whisper_engine.transcribe_chunk,
                            rest_audio,
                            current_language if current_language != "en" else None,
                            "translate" if current_language == "en" else "transcribe"
                        )
                    if rest_result.get("success") and rest_result.get("text"):
                        final_text = (final_text + " " + rest_result["text"]).strip()

            logger.info(f"✅ Streaming complete: \"{final_text[:100]}...\"" if len(final_text) > 100 else f"✅ Streaming complete: \"{final_text}\"")

            return {
                "status": "success",
                "text": final_text,
                "language": current_language or "auto",
                "duration": 0,  # Not easily available in streaming mode
                "streaming": True
            }

        else:
            # --- BATCH FALLBACK: transcribe all audio at once ---
            logger.info("📼 Using batch fallback — transcribing full audio...")

            # Build full audio from any remaining chunks
            all_chunks = list(streaming_buffer)
            if final_chunk is not None:
                all_chunks.append(final_chunk)

            audio_data = np.concatenate(all_chunks, axis=0) if all_chunks else None
            if audio_data is None or len(audio_data) == 0:
                logger.warning("No audio captured")
                return {
                    "status": "success",
                    "text": "",
                    "message": "No audio recorded"
                }

            logger.info(f"📼 Captured {len(audio_data) / 16000:.1f} seconds of audio")

            if not whisper_engine.is_loaded:
                global is_model_loading, model_loading_info
                is_model_loading = True
                model_loading_info = {
                    "model": whisper_engine.model_size if whisper_engine else "default",
                    "status": "Loading model..."
                }
                logger.info("📥 Loading Whisper model...")
                try:
                    success = await loop.run_in_executor(None, whisper_engine.load_model)
                    if not success:
                        is_model_loading = False
                        return {"status": "error", "message": "Failed to load Whisper model"}
                finally:
                    is_model_loading = False
                    model_loading_info = {"model": "", "status": ""}

            logger.info("🎙️ Transcribing full recording (batch)...")

            if current_language == "en":
                task = "translate"
                whisper_language = None
            else:
                task = "transcribe"
                whisper_language = current_language

            transcription_start = time.time()
            result = await loop.run_in_executor(
                None,
                whisper_engine.transcribe_audio,
                audio_data,
                whisper_language,
                task
            )
            transcription_time = time.time() - transcription_start
            logger.info(f"⏱️ Batch transcription took: {transcription_time:.2f}s")

            if not result["success"]:
                return {
                    "status": "error",
                    "message": result.get('error', 'Transcription failed')
                }

            final_text = result["text"].strip()
            logger.info(f"✅ Batch complete: \"{final_text[:100]}...\"" if len(final_text) > 100 else f"✅ Batch complete: \"{final_text}\"")

            return {
                "status": "success",
                "text": final_text,
                "language": result.get("language", "en"),
                "duration": len(audio_data) / 16000,
                "transcription_time": transcription_time,
                "model": whisper_engine.model_size if whisper_engine else "default",
                "device": whisper_engine.device if whisper_engine else "unknown",
                "streaming": False
            }

    except Exception as e:
        logger.error(f"❌ Failed to stop/transcribe: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {"status": "error", "message": str(e)}


@app.post("/cancel")
async def cancel_recording():
    """Cancel recording without transcribing"""
    global is_recording, audio_capture
    global streaming_task, model_load_task, partial_transcript, streaming_buffer, streaming_failed

    try:
        if not is_recording:
            return {"status": "error", "message": "Not recording"}

        logger.info("❌ Canceling recording...")

        is_recording = False

        if streaming_task:
            streaming_task.cancel()
            try:
                await asyncio.wait_for(streaming_task, timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            streaming_task = None

        if model_load_task:
            model_load_task.cancel()
            try:
                await asyncio.wait_for(model_load_task, timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            model_load_task = None

        if audio_capture:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, audio_capture.stop_recording)

        partial_transcript = ""
        streaming_buffer = []
        streaming_failed = False

        logger.info("✅ Recording canceled")

        return {
            "status": "success",
            "message": "Recording canceled"
        }

    except Exception as e:
        logger.error(f"❌ Failed to cancel: {e}")
        return {"status": "error", "message": str(e)}


@app.get("/audio_level")
async def get_audio_level():
    """Get current audio input level (0.0 to 1.0)"""
    global is_recording, audio_capture

    try:
        if not is_recording or not audio_capture:
            return {"level": 0.0, "recording": False}

        # Peek at recent audio without removing from queue
        if audio_capture.audio_queue.empty():
            return {"level": 0.0, "recording": True}

        # Get queue size to know how much audio we have
        queue_size = audio_capture.audio_queue.qsize()

        # Sample a few recent chunks to calculate level
        chunks = []
        temp_chunks = []

        # Get up to 5 most recent chunks
        for _ in range(min(5, queue_size)):
            try:
                chunk = audio_capture.audio_queue.get_nowait()
                temp_chunks.append(chunk)
                chunks.append(chunk)
            except:
                break

        # Put them back in the queue
        for chunk in temp_chunks:
            audio_capture.audio_queue.put(chunk)

        if not chunks:
            return {"level": 0.0, "recording": True}

        # Calculate RMS level
        audio_data = np.concatenate(chunks, axis=0)
        rms = np.sqrt(np.mean(audio_data ** 2))

        # Normalize to 0-1 range (typical speech is around 0.1-0.3 RMS)
        normalized_level = min(1.0, rms * 3.0)

        return {
            "level": float(normalized_level),
            "recording": True,
            "queue_size": queue_size
        }

    except Exception as e:
        logger.error(f"Error getting audio level: {e}")
        return {"level": 0.0, "recording": False, "error": str(e)}


# Removed /get_live_chunk endpoint - using simple record/stop flow now


@app.get("/devices")
async def list_devices():
    """List available audio devices"""
    try:
        import sounddevice as sd
        devices = sd.query_devices()

        input_devices = []
        output_devices = []

        for i, dev in enumerate(devices):
            device_info = {
                "id": i,
                "name": dev["name"],
                "channels": dev["max_input_channels"] if dev["max_input_channels"] > 0 else dev["max_output_channels"],
                "sample_rate": int(dev["default_samplerate"])
            }

            if dev["max_input_channels"] > 0:
                input_devices.append(device_info)
            if dev["max_output_channels"] > 0:
                output_devices.append(device_info)

        return {
            "success": True,
            "inputs": input_devices,
            "outputs": output_devices
        }

    except Exception as e:
        logger.error(f"Error listing devices: {e}")
        return {
            "success": False,
            "error": str(e),
            "inputs": [],
            "outputs": []
        }


@app.get("/gpu/info")
async def get_gpu_info():
    """Get GPU and library installation status"""
    try:
        info = gpu_manager.get_gpu_info()
        return {
            "success": True,
            **info
        }
    except Exception as e:
        logger.error(f"Error getting GPU info: {e}")
        return {
            "success": False,
            "error": str(e)
        }


@app.post("/gpu/install")
async def install_gpu_libs():
    """Download and install GPU libraries (blocking operation)"""
    try:
        logger.info("🚀 Starting GPU library installation...")

        # Run installation synchronously (this will take a while)
        loop = asyncio.get_event_loop()
        success = await loop.run_in_executor(
            None,
            gpu_manager.install_gpu_libs
        )

        if success:
            logger.info("✅ GPU libraries installed successfully")
            return {
                "success": True,
                "message": "GPU libraries installed successfully. Restart may be required."
            }
        else:
            logger.error("❌ GPU library installation failed")
            return {
                "success": False,
                "error": "Installation failed. Check logs for details."
            }

    except Exception as e:
        logger.error(f"❌ GPU installation error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "success": False,
            "error": str(e)
        }


@app.post("/gpu/uninstall")
async def uninstall_gpu_libs():
    """Remove installed GPU libraries"""
    try:
        success = gpu_manager.uninstall_gpu_libs()
        if success:
            return {
                "success": True,
                "message": "GPU libraries removed successfully"
            }
        else:
            return {
                "success": False,
                "error": "Failed to remove GPU libraries"
            }
    except Exception as e:
        logger.error(f"Error uninstalling GPU libs: {e}")
        return {
            "success": False,
            "error": str(e)
        }


# Run the server
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host=BACKEND_HOST,
        port=BACKEND_PORT,
        log_level="info"
    )
