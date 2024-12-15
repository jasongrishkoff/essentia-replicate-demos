# Prediction interface for Cog ⚙️
# Reference: https://github.com/replicate/cog/blob/main/docs/python.md

import json
import tempfile
import requests
import random
from itertools import chain
from pathlib import Path
from textwrap import wrap
import time

import numpy as np
import matplotlib.pyplot as plt
import pandas
import seaborn as sns
from yt_dlp import YoutubeDL
from cog import BasePredictor, Input, Path
from essentia.standard import (
    MonoLoader,
    TensorflowPredictMAEST,
)

from labels import labels

def process_labels(label):
    genre, style = label.split("---")
    return f"{style}\n({genre})"

processed_labels = list(map(process_labels, labels))

def fetch_proxy():
    url = 'https://api.submithub.com/api/proxy'
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        data = response.json()

        if not data:
            raise ValueError("No proxy data returned")

        # Extract the relevant information from the response
        proxy = {
            'username': data['username'],
            'password': data['password'],
            'proxy_address': data['proxy_address'],
            'port': data['ports']['http']  # Using the HTTP port
        }

        return proxy

    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return None
    except (KeyError, ValueError) as e:
        print(f"Error processing proxy data: {e}")
        return None

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory and create the Essentia network for predictions"""
        self.embedding_model_file = "/models/discogs-maest-30s-pw-519l-1.pb"
        self.output = "activations"
        self.sample_rate = 16000

        self.loader = MonoLoader()
        print("attempting to load MAEST")
        self.tensorflowPredictMAEST = TensorflowPredictMAEST(
            graphFilename=self.embedding_model_file, output="StatefulPartitionedCall:0"
        )
        print("loaded MAEST")

    def predict(
        self,
        audio: Path = Input(
            description="Audio file to process",
            default=None,
        ),
        url: str = Input(
            description="YouTube URL to process (overrides audio input)",
            default=None,
        ),
        top_n: int = Input(description="Top n music styles to show", default=10),
        output_format: str = Input(
            description="Output either a bar chart visualization or a JSON blob",
            default="Visualization",
            choices=["Visualization", "JSON"],
        ),
    ) -> Path:
        """Run a single prediction on the model"""

        assert audio or url, "Specify either an audio filename or a YouTube url"

        # If there is a YouTube url use that.
        if url:
            if audio:
                print(
                    "Warning: Both `url` and `audio` inputs were specified. "
                    "The `url` will be process. To process the `audio` input clear the `url` input field."
                )
            audio, title = self._download_with_retry(url)
        else:
            title = audio.name

        print("loading audio...")
        self.loader.configure(
            sampleRate=self.sample_rate,
            resampleQuality=4,
            filename=str(audio),
        )
        waveform = self.loader()

        print("running the model...")
        try:
            activations = self.tensorflowPredictMAEST(waveform)
            activations = np.squeeze(activations)

            # Handle different activation shapes
            if len(activations.shape) == 0:
                # Single value case
                activations_mean = np.array([activations])
            elif len(activations.shape) == 1:
                # Already a 1D array
                activations_mean = activations
            elif len(activations.shape) == 2:
                # 2D array case
                activations_mean = np.mean(activations, axis=0)
            else:
                raise ValueError(f"Unexpected activation shape: {activations.shape}")

            # Ensure we have the correct shape for processing
            if activations_mean.size == 0:
                raise ValueError("No activations were generated - audio may be too short")

            if output_format == "JSON":
                out_path = Path(tempfile.mkdtemp()) / "out.json"
                with open(out_path, "w") as f:
                    json.dump(dict(zip(labels, activations_mean.tolist())), f)
                return out_path

            print("plotting...")
            # Ensure we don't try to get more top_n than we have activations
            top_n = min(top_n, len(activations_mean))
            top_n_idx = np.argsort(activations_mean)[::-1][:top_n]

            # Handle the case where activations is 1D
            if len(activations.shape) == 1:
                activations = activations.reshape(1, -1)

            result = {
                "label": list(
                    chain(
                        *[
                            [processed_labels[idx]] * activations.shape[0]
                            for idx in top_n_idx
                        ]
                    )
                ),
                "activation": list(chain(*[activations[:, idx] for idx in top_n_idx])),
            }
            result = pandas.DataFrame.from_dict(result)

            # Wrap title to lines of approximately 50 chars.
            title = wrap(title, width=50)

            # Allow a maximum of 2 lines of title.
            if len(title) > 2:
                title = title[:2]
                title[-1] += "..."

            title = "\n".join(title)

            g = sns.catplot(
                data=result,
                kind="bar",
                y="label",
                x="activation",
                color="#abc9ea",
                alpha=0.8,
                height=6,
            )
            g.set(xlabel=None)
            g.set(ylabel=None)
            g.set(title=title)
            g.set(xlim=(0, 1))

            # Add some margin so that the title is not cut.
            g.fig.subplots_adjust(top=0.90)

            out_path = Path(tempfile.mkdtemp()) / "out.png"
            plt.savefig(out_path)

            # Clean-up.
            if url:
                audio.unlink()

            print("done!")
            return out_path

        except Exception as e:
            print(f"Error processing audio: {str(e)}")
            if url and audio.exists():
                audio.unlink()
            raise

    def _download_with_retry(self, url, ext="wav", max_retries=2):
        """Attempt to download a YouTube URL with retries"""
        for attempt in range(max_retries + 1):
            try:
                return self._download(url, ext)
            except Exception as e:
                if attempt < max_retries:
                    print(f"Download attempt {attempt + 1} failed: {str(e)}. Retrying...")
                    time.sleep(5)  # Wait for 5 seconds before retrying
                else:
                    print(f"All download attempts failed for URL: {url}")
                    raise

    def _download(self, url, ext="wav"):
        """Download a YouTube URL in the specified format to a temporary directory using a dynamically fetched proxy"""

        proxy = fetch_proxy()
        if not proxy:
            print("Failed to fetch proxy. Exiting...")
            return None, None

        proxy_url = f"http://{proxy['username']}:{proxy['password']}@{proxy['proxy_address']}:{proxy['port']}"

        tmp_dir = Path(tempfile.mktemp())
        ydl_opts = {
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": ext,
            }],
            "postprocessor_args": ["-ar", f"{self.sample_rate}"],
            "outtmpl": str(tmp_dir / f"audio.%(ext)s"),
            "proxy": proxy_url,
        }

        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)

            if "title" in info:
                title = info['title']
            else:
                title = ""  # handle cases where the title might be unavailable

        paths = list(tmp_dir.glob(f"audio.{ext}"))
        assert len(paths) == 1, "Unexpected error: More than one file found!"

        return paths[0], title
