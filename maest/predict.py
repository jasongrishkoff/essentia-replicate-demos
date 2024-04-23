# Prediction interface for Cog ⚙️
# Reference: https://github.com/replicate/cog/blob/main/docs/python.md

import json
import tempfile
import requests
import random
from itertools import chain
from pathlib import Path
from textwrap import wrap

import numpy as np
import matplotlib.pyplot as plt
import pandas
import seaborn as sns
#import youtube_dl
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
    url = 'https://proxy.webshare.io/api/v2/proxy/list/?mode=direct&page=1&page_size=250'
    headers = {'Authorization': f'Token fd9e64adac30d6f46be5ad88b19fffbc42027418'}
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        data = response.json()
        proxies = data.get('results', [])

        if not proxies:
            raise ValueError("No proxies found")

        random_proxy = random.choice(proxies)
        return random_proxy

    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return None


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory and create the Essentia network for predictions"""
        #self.embedding_model_file = "/models/discogs-maest-10s-pw-1.pb"
        #self.embedding_model_file = "/models/discogs-maest-20s-pw-1.pb"
        self.embedding_model_file = "/models/discogs-maest-30s-pw-1.pb"
        #self.embedding_model_file = "/models/discogs-maest-5s-pw-1.pb"
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
            audio, title = self._download(url)
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
        activations = self.tensorflowPredictMAEST(waveform)
        activations = np.squeeze(activations)
        if len(activations.shape) == 2:
            activations_mean = np.mean(activations, axis=0)

        if output_format == "JSON":
            out_path = Path(tempfile.mkdtemp()) / "out.json"
            with open(out_path, "w") as f:
                json.dump(dict(zip(labels, activations_mean.tolist())), f)
            return out_path

        print("plotting...")
        top_n_idx = np.argsort(activations_mean)[::-1][:top_n]

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

    def _download(self, url, ext="wav"):
        """Download a YouTube URL in the specified format to a temporary directory using a dynamically fetched proxy"""

        proxy = fetch_proxy()
        if not proxy:
            print("Failed to fetch proxy. Exiting...")
            return None, None

        proxy_url = f"http://{proxy['username']}:{proxy['password']}@{proxy['proxy_address']}:{proxy['port']}"

        tmp_dir = Path(tempfile.mktemp())
        ydl_opts = {
            "format": "251",
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": ext,
            }],
            "postprocessor_args": ["-ar", f"{self.sample_rate}"],
            "outtmpl": str(tmp_dir / f"audio.%(ext)s"),
            "proxy": proxy_url,  # Set proxy here
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
