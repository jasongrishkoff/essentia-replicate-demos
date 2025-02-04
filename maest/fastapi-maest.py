from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tempfile
from pathlib import Path
import numpy as np
from essentia.standard import MonoLoader, TensorflowPredictMAEST
import json
import shutil
import logging
import uvicorn
from typing import Optional
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

app = FastAPI(title="MAEST Prediction Service")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MAESTPredictor:
    def __init__(self):
        self.embedding_model_file = "/workspace/maest/models/discogs-maest-30s-pw-519l-1.pb"
        self.sample_rate = 16000
        self.loader = None
        self.tensorflowPredictMAEST = None

    def setup(self):
        """Load the model into memory"""
        if not Path(self.embedding_model_file).exists():
            raise RuntimeError(f"Model file not found at {self.embedding_model_file}")
        
        self.loader = MonoLoader()
        logging.info("Loading MAEST model...")
        self.tensorflowPredictMAEST = TensorflowPredictMAEST(
            graphFilename=self.embedding_model_file,
            output="StatefulPartitionedCall:0",
        )
        logging.info("MAEST model loaded successfully")

    async def predict(self, audio_path: str) -> dict:
        """Run prediction on an audio file"""
        if not any(str(audio_path).lower().endswith(ext) for ext in ['.mp3', '.wav']):
            raise ValueError("Only MP3 and WAV files are supported")

        logging.info(f"Loading audio from {audio_path}")
        self.loader.configure(
            sampleRate=self.sample_rate,
            resampleQuality=3,
            filename=str(audio_path),
        )
        waveform = self.loader()

        logging.info("Running model inference...")
        try:
            activations = self.tensorflowPredictMAEST(waveform)
            activations = np.squeeze(activations)

            # Handle different activation shapes
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

            # Load labels
            with open("/workspace/maest/labels.py", "r") as f:
                labels = eval(f.read())

            return dict(zip(labels, activations_mean.tolist()))

        except Exception as e:
            logging.error(f"Error processing audio: {str(e)}")
            raise

predictor = MAESTPredictor()

@app.get("/maest-test")
async def test():
    """Basic health check for MAEST service"""
    return {"status": "online", "service": "MAEST Prediction API"}

@app.on_event("startup")
async def startup_event():
    """Initialize the MAEST model on startup"""
    try:
        predictor.setup()
    except Exception as e:
        logging.error(f"Failed to initialize MAEST model: {e}")
        raise

@app.post("/maest-predict")  # Changed from /predict to /maest-predict to avoid potential conflicts
async def predict(file: UploadFile):
    """Process an audio file and return MAEST predictions"""
    if not file:
        raise HTTPException(status_code=400, detail="No file provided")

    temp_dir = None
    try:
        # Create temporary directory for file processing
        temp_dir = tempfile.mkdtemp()
        temp_file_path = Path(temp_dir) / file.filename

        # Save uploaded file
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Run prediction
        results = await predictor.predict(str(temp_file_path))
        return results

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    finally:
        # Clean up temporary files
        if temp_dir and Path(temp_dir).exists():
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=4000)
