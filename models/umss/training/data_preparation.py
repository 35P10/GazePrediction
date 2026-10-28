"""Training-only EFD generation using the verified original notebook functions."""
from functools import lru_cache
import io
import os
from types import SimpleNamespace

import numpy as np
from PIL import Image

if __package__ == 'training':  # Standalone training Docker image.
    from reference.setup_umss import load_definitions
else:
    from ..reference.setup_umss import load_definitions


def _annotation_csv(path, **kwargs):
    """Repair unused descriptions containing commas, retaining IDs and coordinates."""
    import pandas as pd
    lines = []
    with open(path) as stream:
        for line in stream:
            parts = line.rstrip('\n').split(',')
            if len(parts) > 4:
                parts = [parts[0], ';'.join(parts[1:-2])] + parts[-2:]
            lines.append(','.join(parts) + '\n')
    return pd.read_csv(io.StringIO(''.join(lines)), **kwargs)


@lru_cache(maxsize=None)
def _efd_functions(repository, annotations):
    import pandas as pd
    from shapely.geometry import Point, Polygon
    from skimage.draw import polygon
    # The original parser uses pandas only for these two operations. Its source
    # stays unchanged; the adapter repairs vis386's unused description field.
    pandas_adapter = SimpleNamespace(DataFrame=pd.DataFrame, read_csv=_annotation_csv)
    return load_definitions(
        repository, 'Code/scripts/UMSS/generate_Massvis-EFD.ipynb',
        ('parse_element_label', 'combine_rows', 'polygon_area', 'BBox', 'make_efdmap'),
        {'np': np, 'pd': pandas_adapter, 'Image': Image, 'os': os,
         'Point': Point, 'Polygon': Polygon, 'polygon': polygon, 'annots': annotations})


def make_efdmap(repository, width, height, coords, name, annotations):
    """Execute the original notebook's EFD definitions with a local annotation path."""
    functions = _efd_functions(str(repository), str(annotations))
    with np.errstate(invalid='ignore', divide='ignore'):
        return functions['make_efdmap'](width, height, coords, name)
