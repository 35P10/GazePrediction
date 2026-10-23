import numpy as np

from .model import IMAGE_MEAN, IMAGE_STD, VisSalFormer


class VisSalFormerUtils:
    """Inspect inputs using an existing model; no additional weights are loaded."""

    def __init__(self, model: VisSalFormer):
        self.model = model

    def inputs(self, image_path, question):
        """``encode`` in readable form: the image as the network sees it (de-normalized for display) and the tokens."""
        from PIL import Image

        pixel_values, text = self.model.encode(image_path, question)
        mean, std = np.array(IMAGE_MEAN), np.array(IMAGE_STD)
        with Image.open(image_path) as source:
            chart = np.asarray(source.convert('RGB'))
        return {
            'chart (original)': chart,
            'image given to the network': (pixel_values[0].numpy().transpose(1, 2, 0) * std + mean).clip(0, 1),
            'original size (width × height)': f'{chart.shape[1]} × {chart.shape[0]} px',
            'question': question,
            'tokens': self.model.tokenizer.convert_ids_to_tokens(text['input_ids'][0]),
        }


def chart_probability(saliency, width, height):
    """Resize a raw map to the chart size (bilinear) and scale it to sum to 1."""
    import cv2

    probability = cv2.resize(saliency.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
    return probability / probability.sum()

