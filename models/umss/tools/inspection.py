"""Inspect and visualize UMSS preprocessing without running the neural model."""
from ..sampling import PACKAGE_ROOT

PREPROCESS_SOURCE = PACKAGE_ROOT.parents[1] / 'reproductions/umss_massvis/official_repository/Code/scripts/saliency/sal_imp_utilities.py'


class UMSSUtils:
    """Inspect preprocessing without importing TensorFlow or running the network."""

    @staticmethod
    def inputs(image_path):
        import ast
        import cv2
        import numpy as np
        # Execute only the two pure upstream helpers; importing their module would load Keras.
        from ..reference.setup_umss import verify_source_file
        verify_source_file(PREPROCESS_SOURCE, 'Code/scripts/saliency/sal_imp_utilities.py')
        tree = ast.parse(PREPROCESS_SOURCE.read_text())
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name in ('padding', 'preprocess_images')]
        namespace = dict(np=np, cv2=cv2)
        module = ast.Module(body=functions, type_ignores=[])
        exec(compile(module, 'UMSS upstream preprocessing', 'exec'), namespace)
        tensor = namespace['preprocess_images']([str(image_path)], 240, 320)
        rgb = np.clip(tensor[0] + np.array([103.939,116.779,123.68]),0,255).astype(np.uint8)[...,::-1]
        return {'model image': rgb, 'tensor': tensor,
                'preprocessing': 'upstream BGR, centered zero letterbox, channel means subtracted'}
