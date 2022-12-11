import os

os.system("python -m pip install pyyaml==5.1")
os.system("python -m pip install fvcore")
import os, distutils.core
os.system("git clone 'https://github.com/facebookresearch/detectron2'")
os.system("python -m pip install -e detectron2")
dist = distutils.core.run_setup("./detectron2/setup.py")


import torch, detectron2

TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
print("detectron2:", detectron2.__version__)
