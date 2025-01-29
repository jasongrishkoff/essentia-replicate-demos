import json
import tempfile
from pathlib import Path
import numpy as np
from cog import BasePredictor, Input, Path
from essentia.standard import (
    MonoLoader,
    TensorflowPredictMAEST,
)

from labels import labels

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory and create the Essentia network for predictions"""
        self.embedding_model_file = "/models/discogs-maest-30s-pw-519l-1.pb"
        self.sample_rate = 16000

        self.loader = MonoLoader()
        print("attempting to load MAEST")
        self.tensorflowPredictMAEST = TensorflowPredictMAEST(
            graphFilename=self.embedding_model_file,
            output="StatefulPartitionedCall:0",
            # Default is 1875, increase to reduce overlap and processing time
            # patchHopSize=2000,  # This will halve the number of patches processed
            # batchSize=-1  # Uncomment to accumulate all patches for single processing
        )
        print("loaded MAEST")

    def predict(
        self,
        audio: Path = Input(
            description="MP3 file to process",
            default=None,
        ),
        url: str = Input(
            description="[DEPRECATED] YouTube URLs are no longer supported",
            default=None,
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        if url:
            raise ValueError("YouTube URL processing is no longer supported. Please provide an MP3 file.")
        
        if not audio:
            raise ValueError("No audio file provided")

        if not any(str(audio).lower().endswith(ext) for ext in ['.mp3', '.wav']):
            raise ValueError("Only MP3 and WAV files are supported")

        print("loading audio...")
        self.loader.configure(
            sampleRate=self.sample_rate,
            resampleQuality=3,
            filename=str(audio),
        )
        waveform = self.loader()

        print("running the model...")
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

            out_path = Path(tempfile.mkdtemp()) / "out.json"
            with open(out_path, "w") as f:
                json.dump(dict(zip(labels, activations_mean.tolist())), f)
            
            return out_path

        except Exception as e:
            print(f"Error processing audio: {str(e)}")
            raise
