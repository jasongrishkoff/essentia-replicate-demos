FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

WORKDIR /app

RUN apt-get update && apt-get install -y \
    software-properties-common \
    tzdata \
    nvidia-cuda-toolkit \
    nvidia-cuda-toolkit-gcc \
    ffmpeg \
    curl \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN python3.11 -m pip install --upgrade pip

ENV TF_CUDA_COMPUTE_CAPABILITIES="7.0,7.5,8.0,8.6,8.9,9.0"
ENV TF_CUDA_VERSION=11.8
ENV TF_CUDNN_VERSION=8
ENV XLA_FLAGS="--xla_gpu_cuda_dir=/usr/local/cuda"

RUN pip3 install --no-cache-dir \
    numpy==1.26.4 \
    tensorflow==2.15.0 \
    essentia-tensorflow==2.1b6.dev1110 \
    runpod \
    requests

# Create model directory and download files
RUN mkdir -p /app/models && \
    curl -o /app/models/discogs-maest-30s-pw-519l-1.pb https://submithub.s3.us-east-1.amazonaws.com/models/discogs-maest-30s-pw-519l-1.pb && \
    curl -o /app/labels.py https://submithub.s3.us-east-1.amazonaws.com/models/labels.py

RUN python3.11 -c "import tensorflow as tf; \
    tf.config.experimental.enable_mlir_bridge(); \
    tf.config.optimizer.set_jit(True)"

COPY <<'EOF' /app/predictor.py
import numpy as np
from pathlib import Path
import logging
import subprocess
from essentia.standard import MonoLoader, TensorflowPredictMAEST

def verify_audio_file(file_path):
    try:
        result = subprocess.run([
            'ffprobe', '-v', 'error',
            '-select_streams', 'a:0',
            '-show_entries', 'stream=codec_name',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            file_path
        ], capture_output=True, text=True)
        return result.returncode == 0
    except Exception as e:
        logging.error(f"Error verifying audio file: {str(e)}")
        return False

_maest_instance = None

def get_maest():
    global _maest_instance
    if _maest_instance is None:
        _maest_instance = TensorflowPredictMAEST(
            graphFilename="/app/models/discogs-maest-30s-pw-519l-1.pb",
            output="StatefulPartitionedCall:0",
        )
    return _maest_instance

class Predictor:
    def setup(self):
        self.sample_rate = 16000
        self.loader = MonoLoader()
        self.tensorflowPredictMAEST = get_maest()
        
        with open("/app/labels.py", "r") as f:
            exec(f.read(), globals())
        self.labels = labels

    def predict(self, audio_path):
        if not any(str(audio_path).lower().endswith(ext) for ext in ['.mp3', '.wav']):
            raise ValueError("Only MP3 and WAV files are supported")

        if not verify_audio_file(audio_path):
            raise ValueError("Invalid or corrupted audio file")

        self.loader.configure(
            sampleRate=self.sample_rate,
            resampleQuality=3,
            filename=str(audio_path),
        )
        waveform = self.loader()

        try:
            activations = self.tensorflowPredictMAEST(waveform)
            activations = np.squeeze(activations)

            if len(activations.shape) == 0:
                activations_mean = np.array([activations])
            elif len(activations.shape) == 1:
                activations_mean = activations
            elif len(activations.shape) == 2:
                activations_mean = np.mean(activations, axis=0)
            else:
                raise ValueError(f"Unexpected activation shape: {activations.shape}")

            if activations_mean.size == 0:
                raise ValueError("No activations were generated - audio may be too short")

            return dict(zip(self.labels, activations_mean.tolist()))

        except Exception as e:
            logging.error(f"Error processing audio: {str(e)}")
            raise
EOF

COPY <<'EOF' /app/handler.py
import runpod
import tempfile
import shutil
import os
from pathlib import Path
import requests
import logging
import time
import subprocess
from predictor import Predictor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def log_timing(start_time, step_name):
    duration = time.time() - start_time
    logging.info(f"🕒 {step_name} took {duration:.2f} seconds")

def verify_audio_file(file_path):
    """Verify audio file is valid using ffprobe."""
    try:
        result = subprocess.run([
            'ffprobe', '-v', 'error',
            '-select_streams', 'a:0',
            '-show_entries', 'stream=codec_name',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            file_path
        ], capture_output=True, text=True)
        return result.returncode == 0
    except Exception as e:
        logging.error(f"Error verifying audio file: {str(e)}")
        return False

def download_audio_file(url: str) -> tuple[Path, str]:
    download_start = time.time()
    temp_dir = None
    try:
        temp_dir = tempfile.mkdtemp()
        os.chmod(temp_dir, 0o755)

        url_path = requests.utils.urlparse(url).path.lower()
        if url_path.endswith('.wav'):
            ext = '.wav'
        else:
            ext = '.mp3'

        temp_file_path = os.path.join(temp_dir, f'audio{ext}')
        
        logging.info(f"Downloading file from URL: {url}")
        response = requests.get(url, stream=True, timeout=90)
        response.raise_for_status()

        content_type = response.headers.get('content-type', '').lower()
        if 'wav' in content_type and ext != '.wav':
            temp_file_path = os.path.join(temp_dir, 'audio.wav')
            logging.info("Switched to .wav extension based on Content-Type")

        content_length = response.headers.get('content-length')
        if content_length:
            logging.info(f"File size: {int(content_length) / 1024 / 1024:.2f} MB")

        with open(temp_file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        logging.info(f"File downloaded to {temp_file_path}")

        if not verify_audio_file(temp_file_path):
            raise ValueError("Invalid audio file")

        log_timing(download_start, "File download")
        return temp_file_path, temp_dir

    except Exception as e:
        if temp_dir and Path(temp_dir).exists():
            shutil.rmtree(temp_dir)
        raise Exception(f"Failed to download audio file: {str(e)}")

def handler(job):
    total_start = time.time()
    temp_dir = None
    try:
        setup_start = time.time()
        predictor = Predictor()
        predictor.setup()
        log_timing(setup_start, "Model setup")
        
        job_input = job["input"]
        
        if "audio" not in job_input:
            return {"error": "No audio URL provided"}
            
        audio_path, temp_dir = download_audio_file(job_input["audio"])
        
        predict_start = time.time()
        result = predictor.predict(audio_path)
        log_timing(predict_start, "Model prediction")
        
        log_timing(total_start, "Total processing")
        return {"status": "success", "output": result}
        
    except Exception as e:
        logging.error(f"Error in handler: {str(e)}")
        return {"status": "error", "error": str(e)}
    finally:
        if temp_dir and Path(temp_dir).exists():
            shutil.rmtree(temp_dir)

runpod.serverless.start({"handler": handler})
EOF

ENV CUDA_VISIBLE_DEVICES=0
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

CMD ["python3.11", "-u", "handler.py"]
