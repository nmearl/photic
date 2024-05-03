import numpy as np
from .datasets import generate_mock_data
import torch
import neuralprocesses.torch as nps
from neuralprocesses.model import Model
import os
import lab as B
import dill


def predict(model: Model | str, device: str):
    device = torch.device("cpu")

    if device == "gpu":
        if torch.cuda.is_available():
            print("Setting device to the gpu to `cuda`")
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            print("Setting device to the gpu to `mps`")
            device = torch.device("mps")

            # Required or else the we may run out of memory
            os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
        else:
            print("Failed to find a GPU, falling back to CPU.")

    B.set_global_device(device)
    torch.set_default_dtype(torch.float32)

    if isinstance(model, str):
        with open(model, 'rb') as f:
            model = dill.load(f)

    # Make predictions on some new data
    pxt, pyt = generate_mock_data(
        amplitude_range=(0.9, 1.1),
        x_break_range=(95, 105),
        alpha_1_range=(-2.5, -1.5),
        alpha_2_range=(1.5, 2.25),
        delta_range=(0.09, 1.1),
        batch_size=1, num_points=100, 
        device=device)

    # Use batch to create random set of context points
    maxctx = 100
    inds = np.random.permutation(pxt.shape[2])
    pxc, pyc = pxt[:1, :, inds[:maxctx]], pyt[:1, :, inds[:maxctx]]
    pxc, pyc = pxt[:1, :, :maxctx], pyt[:1, :, :maxctx]

    # Running without moving back to the cpu, or simply using `ar_predict` either crashes
    # the system or runs out of memory...
    # with B.on_device('cpu'):
    #     mean, var, noiseless_samples, noisy_samples = nps.predict(convcnp.cpu(), pxc.cpu(), pyc.cpu(), pxt.cpu())
    with torch.no_grad():
        mean, var, noiseless_samples, noisy_samples = nps.ar_predict(model, pxc, pyc, pxt)

    # mean, var, noiseless_samples, noisy_samples = nps.ar_predict(convcnp, pxc, pyc, pxt)
    return pxc, pyc, pxt, pyt, mean, var, noiseless_samples, noisy_samples
