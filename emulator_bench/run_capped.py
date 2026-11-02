"""Runs a script on the GPU with a cap on the PyTorch memory of the process: run_capped.py SCRIPT [ARGS...].

The cap is DEEPENZYME_GPU_FRACTION of the GPU memory. When the process reaches it, PyTorch frees its cached blocks
before it reports an error, so several runs fit on one GPU.
"""

import os
import runpy
import sys

import torch

torch.cuda.set_per_process_memory_fraction(float(os.environ["DEEPENZYME_GPU_FRACTION"]))
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
