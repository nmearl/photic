import lab as B
import torch
import os
import numpy as np
import neuralprocesses.torch as nps
from photic.datasets import generate_mock_data
import wandb
import dill
from pathlib import Path


def run(
    device: str = "cpu",
    learn_rate: float = 3e-4,
    epochs: float = 1e2,
    save_path: str = f"./model.pkl",
):
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
    B.set_random_seed(0)

    # Start a new wandb run to track this script
    wandb.init(
        # Set the wandb project where this run will be logged
        project="photic",
        # Track hyperparameters and run metadata
        config={
            "learning_rate": learn_rate,
            "architecture": "ConvCNP",
            "dataset": "TDE-MOCK",
            "epochs": epochs,
        },
    )

    # Construct a ConvCNP.
    convcnp = nps.construct_convgnp(
        dim_x=1,
        dim_y=1,
        likelihood="het",
        # dim_lv=16,
        points_per_unit=1,
        dtype=torch.float32,
    )
    convcnp = convcnp.to(device)

    # Construct optimiser.
    opt = torch.optim.Adam(convcnp.parameters(), learn_rate)

    for i in range(epochs):
        # Sample a batch of new context and target sets. Replace this with your data. The
        # shapes are `(batch_size, dimensionality, num_data)`.
        x, y = generate_mock_data(
            amplitude_range=(0.9, 1.1),
            x_break_range=(95, 105),
            alpha_1_range=(-2.5, -1.5),
            alpha_2_range=(1.5, 2.25),
            delta_range=(0.09, 1.1),
            batch_size=32,
            num_points=200,
            device=device,
        )

        # For every batch, flip a coin. If heads, do you what you now do. If tails,
        # choose a point on the x-axis. Every point smaller than that point becomes
        # context, and every point bigger becomes target.
        if np.random.random() < 0.5:
            inds = np.random.permutation(x.shape[2])
            split = [np.random.randint(0, x.shape[2]), 35]
        else:
            inds = np.arange(x.shape[2])
            dem = np.random.randint(1, x.shape[2])
            split = [np.random.randint(0, dem), x.shape[2] - dem]

        xc, yc = x[:, :, inds[: split[0]]], y[:, :, inds[: split[0]]]
        xt, yt = x[:, :, inds[split[1] :]], y[:, :, inds[split[1] :]]

        # Compute the loss and update the model parameters.
        loss = -torch.mean(
            nps.loglik(convcnp, xc, yc, xt, yt, normalise=True, dtype_lik=torch.float32)
        )

        wandb.log({"nctx": split[0], "ntrg": split[1], "loss": loss})

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    # Save out model
    with open(save_path, "wb") as f:
        dill.dump(convcnp, f)

    print(f"Saved model to '{save_path}'.")
