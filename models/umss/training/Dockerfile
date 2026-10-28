# UMSS stage-1 training environment. Build context: models/umss
#   docker build -f training/Dockerfile -t umss-train models/umss
#
# The authors used TensorFlow 1.14 + Keras 2.3.1 (DaRUS requirements.txt). Official TF 1.x
# wheels stop at CUDA 10, which has no kernels for Ampere (RTX 30xx). NVIDIA's TF1 build
# (TF 1.15.5) runs on Ampere/Ada and keeps the same graph-mode Keras API.
# 23.03 needs host driver >= 530; on an older driver use TF1_TAG=22.12-tf1-py3 (>= 520).
ARG TF1_TAG=23.03-tf1-py3
FROM nvcr.io/nvidia/tensorflow:${TF1_TAG}

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=1 MPLBACKEND=Agg
# h5py 2.10 is required by Keras 2.3.1 load_weights (h5py 3 returns str, Keras calls .decode).
# numpy stays at the version NVIDIA built TensorFlow against (>=1.22,<1.24 for nv23.03).
# Read it from Python: `pip freeze` lists NVIDIA's numpy as a file URL, not `numpy==`.
RUN python -c "import numpy; print('numpy==' + numpy.__version__)" > /tmp/constraints.txt \
    && cat /tmp/constraints.txt \
    && pip install --no-cache-dir -c /tmp/constraints.txt \
        Keras==2.3.1 Keras-Applications==1.0.8 Keras-Preprocessing==1.1.2 \
        h5py==2.10.0 \
        opencv-python-headless==4.5.5.64 \
        pandas==1.5.3 shapely==2.0.1 scikit-image==0.19.3 \
        Pillow==9.4.0 matplotlib==3.6.3 natsort==8.3.1 \
    && python -c "import keras, tensorflow as tf; assert keras.__version__ == '2.3.1' and tf.__version__.startswith('1.15')"

WORKDIR /umss
COPY reference/ reference/
COPY evaluation/ evaluation/
COPY training/ training/
ENTRYPOINT ["bash", "/umss/training/pipeline.sh"]
CMD ["all"]
