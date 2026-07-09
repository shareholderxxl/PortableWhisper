"""
GPU Capability Test Script (Legacy — Phase 1 CTranslate2)
Tests if the GPU can handle ONNX Runtime / DirectML operations
"""

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_gpu_info():
    """Test basic GPU information"""
    try:
        import ctranslate2

        logger.info("=" * 60)
        logger.info("CTranslate2 GPU Capability Test")
        logger.info("=" * 60)

        # Get CUDA device count
        cuda_count = ctranslate2.get_cuda_device_count()
        logger.info(f"CUDA devices found: {cuda_count}")

        if cuda_count == 0:
            logger.error("❌ No CUDA devices detected!")
            return False

        # Get device properties for each GPU
        for i in range(cuda_count):
            logger.info(f"\n📊 GPU {i} Details:")
            try:
                # Try to get compute capability
                import torch
                if torch.cuda.is_available():
                    props = torch.cuda.get_device_properties(i)
                    logger.info(f"   Name: {props.name}")
                    logger.info(f"   Compute Capability: {props.major}.{props.minor}")
                    logger.info(f"   Total Memory: {props.total_memory / 1024**3:.1f} GB")
                    logger.info(f"   Multi-Processors: {props.multi_processor_count}")

                    # Check if compute capability is sufficient
                    compute_cap = props.major * 10 + props.minor
                    if compute_cap < 35:
                        logger.error(f"   ❌ Compute Capability {props.major}.{props.minor} is too old!")
                        logger.error(f"   Minimum required: 3.5")
                        return False
                    elif compute_cap < 70:
                        logger.warning(f"   ⚠️ Compute Capability {props.major}.{props.minor} does not support FP16")
                        logger.info(f"   Will use int8 compute type")
                    else:
                        logger.info(f"   ✅ Compute Capability {props.major}.{props.minor} supports all features")
            except ImportError:
                logger.warning("   ⚠️ PyTorch not installed, cannot get detailed GPU info")

        logger.info("\n" + "=" * 60)
        logger.info("Testing CTranslate2 model loading...")
        logger.info("=" * 60)

        # Test if we can create a simple CTranslate2 model
        from faster_whisper import WhisperModel

        # Try loading a tiny model on GPU with int8 (most compatible)
        logger.info("Attempting to load tiny model on CUDA with int8...")
        try:
            model = WhisperModel(
                "tiny",
                device="cuda",
                compute_type="int8"
            )
            logger.info("✅ Model loaded successfully on CUDA!")

            # Try a test transcription with dummy audio
            import numpy as np
            logger.info("\nTesting transcription with 1 second of silence...")
            dummy_audio = np.zeros(16000, dtype=np.float32)

            try:
                segments, info = model.transcribe(dummy_audio, language="en")
                segments_list = list(segments)
                logger.info("✅ Transcription test PASSED!")
                logger.info(f"   Detected language: {info.language}")
                logger.info("   GPU is fully functional for Whisper!")
                return True

            except Exception as trans_error:
                logger.error(f"❌ Transcription FAILED: {trans_error}")
                logger.error("   GPU can load model but cannot execute transcription")
                logger.error("   This may indicate:")
                logger.error("   1. Driver too old for CUDA 12")
                logger.error("   2. Insufficient GPU memory")
                logger.error("   3. Incompatible cuDNN version")
                return False

        except Exception as model_error:
            logger.error(f"❌ Model loading FAILED: {model_error}")
            logger.error("   Check CUDA library installation")
            return False

    except Exception as e:
        logger.error(f"❌ GPU test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_driver_version():
    """Check NVIDIA driver version"""
    try:
        import subprocess
        logger.info("\n" + "=" * 60)
        logger.info("NVIDIA Driver Information")
        logger.info("=" * 60)

        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=driver_version,cuda_version', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode == 0:
            driver_ver, cuda_ver = result.stdout.strip().split(', ')
            logger.info(f"Driver Version: {driver_ver}")
            logger.info(f"CUDA Version (driver max): {cuda_ver}")

            # Check if driver supports CUDA 12
            cuda_major = int(cuda_ver.split('.')[0])
            if cuda_major < 12:
                logger.warning(f"⚠️ Driver only supports CUDA {cuda_ver}")
                logger.warning(f"   CUDA 12.x requires driver >= 525.60.13")
                logger.warning(f"   Update your NVIDIA driver!")
                return False
            else:
                logger.info(f"✅ Driver supports CUDA {cuda_ver}")
                return True
        else:
            logger.warning("Could not query nvidia-smi")
            return False

    except FileNotFoundError:
        logger.error("❌ nvidia-smi not found")
        logger.error("   NVIDIA drivers may not be installed properly")
        return False
    except Exception as e:
        logger.warning(f"Could not check driver version: {e}")
        return False

if __name__ == "__main__":
    logger.info("\n🔍 Starting GPU Capability Tests...\n")

    # Test 1: Driver version
    driver_ok = test_driver_version()

    # Test 2: GPU info and transcription
    gpu_ok = test_gpu_info()

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Driver Check: {'✅ PASS' if driver_ok else '❌ FAIL'}")
    logger.info(f"GPU Test: {'✅ PASS' if gpu_ok else '❌ FAIL'}")

    if driver_ok and gpu_ok:
        logger.info("\n✅ GPU is fully functional for PortableWhisper!")
    else:
        logger.info("\n❌ GPU has compatibility issues")
        if not driver_ok:
            logger.info("   ACTION: Update NVIDIA driver to latest version")
        if not gpu_ok:
            logger.info("   ACTION: Check error messages above for details")

    logger.info("=" * 60)
