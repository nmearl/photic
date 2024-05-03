import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot(pxc, pyc, pxt, pyt, mean, var, save_path="./out.png"):
    sns.set_style("ticks")
    sns.set_context("talk", font_scale=1.5)

    f, ax = plt.subplots(figsize=(12, 8), layout='constrained')

    for i in range(1):
        ctx_x, ctx_y = pxc.cpu().numpy()[i, 0, :], pyc.cpu().numpy()[i, 0, :]
        tar_x, tar_y = pxt.cpu().numpy()[i, 0, :], pyt.cpu().numpy()[i, 0, :]

        ax.scatter(ctx_x, ctx_y)
        ax.scatter(tar_x, tar_y, color="none", edgecolor="k")

    for i in range(1):
        pre_x, pre_y = pxt.cpu().numpy()[i, 0, :], mean.detach().cpu().numpy()[i, 0, :]
        pre_yerr = np.sqrt(var.detach().cpu().numpy()[i, 0, :])
        
        ax.plot(pre_x, pre_y)
        ax.fill_between(pre_x, pre_y - pre_yerr, pre_y + pre_yerr, alpha=0.25, color="C0")

    ax.set_ylabel("Normalized Flux")
    ax.set_xlabel("Time Since Disruption [MJD]")

    f.savefig(f"{save_path}")
