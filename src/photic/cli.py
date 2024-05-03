import click
from .run import run as run_np
from .plot import plot as plot_np
from .predict import predict as predict_np
from uuid import uuid4


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "--device", type=click.Choice(["cpu", "gpu"], case_sensitive=False), default="cpu"
)
@click.option("--learn-rate", type=float, default=3e-4, help="Learn rate")
@click.option("--epochs", type=int, default=1e2, help="Number of epochs to train over")
@click.option(
    "--save-path",
    type=click.Path(exists=True),
    default=f"./model_{uuid4()}.pkl",
    help="Output file path of saved trained model.",
)
def run(device, learn_rate, epochs, save_path):
    run_np(device, learn_rate, epochs, save_path)


@cli.command()
@click.option(
    "--device", type=click.Choice(["cpu", "gpu"], case_sensitive=False), default="cpu"
)
@click.option(
    "--model-path",
    type=click.Path(exists=True),
    required=True,
    help="Path to trained model.",
)
@click.option(
    "--save-path",
    type=click.Path(),
    default=f"./plot_{uuid4()}.png",
    help="Output file path of image.",
)
def plot(device, model_path, save_path):
    (pxc, pyc, pxt, pyt, mean, var, noiseless_samples, noisy_samples) = predict_np(
        model_path, device
    )
    plot_np(pxc, pyc, pxt, pyt, mean, var, save_path)


if __name__ == "__main__":
    cli()
