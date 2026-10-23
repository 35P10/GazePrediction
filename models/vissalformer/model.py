from __future__ import annotations

import sys
from itertools import islice
from pathlib import Path
from typing import Dict, Iterable, Optional, TypedDict, Union

import numpy as np

VISSALFORMER_DIR = Path(__file__).resolve().parents[2] / "external" / "vissalformer"

IMAGE_MEAN = (0.8801, 0.8827, 0.8840)
IMAGE_STD = (0.2523, 0.2321, 0.2400)

PathLike = Union[str, Path]


class ImageQuestionInput(TypedDict):
    """One inference request; annotations and expected answers are excluded."""
    condition_id: str
    image_path: PathLike
    question: str


def preprocess_image(image_path):
    """Exact image tensor used by ``encode``; also available for dataset readability review without weights."""
    import torch
    from PIL import Image

    with Image.open(image_path) as source:
        image = np.array(source.convert("RGB"), dtype=np.float32).transpose(2, 0, 1) / 255
    pixels = torch.nn.functional.interpolate(
        torch.from_numpy(image).unsqueeze(0), size=(224, 224), mode="bilinear",
        align_corners=False, antialias=True)
    mean = pixels.new_tensor(IMAGE_MEAN).view(1, 3, 1, 1)
    std = pixels.new_tensor(IMAGE_STD).view(1, 3, 1, 1)
    return (pixels - mean) / std


class VisSalFormer:
    """Pretrained VisSalFormer; ``predict`` returns the raw map at the model's resolution."""

    def __init__(self, image_model="microsoft/swin-tiny-patch4-window7-224", text_model="bert-base-uncased"):
        import torch
        from transformers import AutoTokenizer, BertModel, SwinModel
        sys.path.insert(0, str(VISSALFORMER_DIR))  # The authors' salformer.py is not a package.
        from salformer import SalFormer

        self.image_model = image_model
        self.text_model = text_model
        self.tokenizer = AutoTokenizer.from_pretrained(text_model)
        self.network = SalFormer(SwinModel.from_pretrained(image_model), BertModel.from_pretrained(text_model))
        checkpoint = torch.load(VISSALFORMER_DIR / "VisSalFormer_weights.tar", map_location="cpu")
        self.network.load_state_dict(checkpoint["model_state_dict"])
        self.network.eval()

    def encode(self, image_path, question):
        """Image preprocessing matching the published SalChartQA tensors, plus question tokens."""
        return preprocess_image(image_path), self.tokenizer(question, return_tensors="pt")

    def predict(self, image_path: PathLike, question: str) -> np.ndarray:
        """Raw saliency map for one chart image and question."""
        import torch

        pixel_values, text_inputs = self.encode(image_path, question)
        device = next(self.network.parameters()).device
        with torch.no_grad():
            saliency = self.network(pixel_values.to(device),
                                    {key: value.to(device) for key, value in text_inputs.items()})
        return saliency[0, 0].cpu().numpy()

    def run(self, inputs: Iterable[ImageQuestionInput], output_dir: Optional[PathLike] = None,
            *, batch_size: int = 1) -> Union[Path, Dict[str, np.ndarray]]:
        """Prepare PNG inputs and return raw maps keyed by condition ID.

        When ``output_dir`` is supplied, save maps there and return its path instead.

        By default, questions are processed individually. With ``batch_size > 1``,
        preserve input order and pad questions to the longest question in each batch.
        Batch composition affects predictions because text cross-attention does not mask padding.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        predictions = {}
        maps_dir = None
        if output_dir is not None:
            output_dir = Path(output_dir)
            maps_dir = output_dir / 'maps'
            maps_dir.mkdir(parents=True, exist_ok=True)

        def collect(condition_id, saliency):
            if maps_dir is None:
                predictions[condition_id] = saliency
            else:
                np.save(maps_dir / (condition_id + '.npy'), saliency)

        if batch_size == 1:
            for item in inputs:
                saliency = self.predict(item['image_path'], item['question'])
                collect(item['condition_id'], saliency)
            return predictions if output_dir is None else output_dir

        import torch

        device = next(self.network.parameters()).device
        self.network.eval()
        pending = iter(inputs)
        while True:
            batch = list(islice(pending, batch_size))
            if not batch:
                break
            pixels = torch.cat([preprocess_image(item['image_path']) for item in batch])
            questions = self.tokenizer([item['question'] for item in batch], return_tensors='pt', padding=True)
            with torch.no_grad():
                saliency = self.network(pixels.to(device),
                                        {key: value.to(device) for key, value in questions.items()})
            for item, prediction in zip(batch, saliency[:, 0].cpu().numpy()):
                collect(item['condition_id'], prediction)
        return predictions if output_dir is None else output_dir

    def run_with_tensors(self, batches, folder: PathLike = None, *, device=None):
        """Yield ``(saliency, batch)`` for batches from the official SalChartQA collate function.

        Each batch is ``(images, tokenized_questions, fixations, heatmaps, names)``; only the first two
        reach the network, unchanged. ``folder`` receives PNGs saved with the official ``save_image``.
        """
        import torch

        device = next(self.network.parameters()).device if device is None else torch.device(device)
        self.network.to(device).eval()
        if folder is not None:
            from torchvision.utils import save_image

            folder = Path(folder)
            folder.mkdir(parents=True, exist_ok=True)
        for batch in batches:
            images, text_inputs, _, _, names = batch
            with torch.no_grad():
                saliency = self.network(images.to(device), {key: value.to(device)
                                                           for key, value in text_inputs.items()})
            if folder is not None:
                for prediction, name in zip(saliency, names):
                    save_image(prediction, str(folder / Path(str(name)).name))
            yield saliency, batch


def create_predictor(**settings):
    return VisSalFormer(**settings)
