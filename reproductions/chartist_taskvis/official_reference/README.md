# vision-env
human vision env

## Installing

```
conda env create -f env.yml
conda activate vision
```

Update the requirements

```
conda env update -f env.yml
```

Install PyTorch with CUDA

```
pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 --index-url https://download.pytorch.org/whl/cu118
```

## Label Charts

https://github.com/wkentaro/labelme

```
labelme
```

For chart-relevant labels, we need to label "title", "axis", "mark".
For task-relevant labels, we need to label area of "search", "map_mark", "map" (tip of the mark), "read".

```
python data_gen.py
```