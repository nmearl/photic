import lab as B
import torch

import neuralprocesses.torch as nps


def clsconvgnp():
    """
    Attempt to implement classification with convolutional GNP.
    """
    dim_x = 1
    dim_y = 1

    # CNN architecture:
    unet = nps.UNet(
        dim=dim_x,
        in_channels=2 * dim_y,
        # Add an extra channel for classification.
        out_channels=(2 + 512) * dim_y + 1,
        channels=(8, 16, 16, 32, 32, 64),
    )

    # Discretisation of the functional embedding:
    disc = nps.Discretisation(
        points_per_unit=64,
        multiple=2**unet.num_halving_layers,
        margin=0.1,
        dim=dim_x,
    )

    # Create the encoder and decoder and construct the model.
    encoder = nps.FunctionalCoder(
        disc,
        nps.Chain(
            nps.PrependDensityChannel(),
            nps.SetConv(scale=1 / disc.points_per_unit),
            nps.DivideByFirstChannel(),
            nps.DeterministicLikelihood(),
        ),
    )
    decoder = nps.Chain(
        unet,
        nps.SetConv(scale=1 / disc.points_per_unit),
        # Besides the regression prediction, also output a tensor of log-probabilities.
        nps.Splitter((2 + 512) * dim_y, 1),
        nps.Parallel(
            nps.LowRankGaussianLikelihood(512),
            lambda x: x,
        ),
    )
    convgnp = nps.Model(encoder, decoder)

    return convgnp
