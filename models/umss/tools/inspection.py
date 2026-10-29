"""Inspect UMSS preprocessing without running the neural model."""
from ..reference.setup_umss import DEFAULT_DESTINATION, load_definitions


class UMSSUtils:
    """Inspect preprocessing without importing TensorFlow or running the network."""

    @staticmethod
    def inputs(image_path):
        from copy import deepcopy
        import cv2
        import numpy as np
        # Official functions, loaded alone: importing their module would load Keras.
        official = load_definitions(DEFAULT_DESTINATION, 'Code/scripts/saliency/sal_imp_utilities.py',
                                    ('padding', 'preprocess_images', 'reverse_preprocess'),
                                    dict(np=np, cv2=cv2, deepcopy=deepcopy))
        tensor = official['preprocess_images']([str(image_path)], 240, 320)
        return {'model image': official['reverse_preprocess'](tensor[0]), 'tensor': tensor,
                'preprocessing': 'upstream BGR, centered zero letterbox, channel means subtracted'}
