import matplotlib.pyplot as plt
import numpy as np


N = 5000
z = 3
steps = 5
stride = 1 / steps
t = np.arange(0, 1 + 1e-9, stride)

rng = np.random.default_rng(0)

for i in t:
    alpha = i
    beta = 1 - i

    # 2D Gaussian: X ~ N(mu, Sigma)
    mu = np.array([alpha * z, alpha * z], dtype=float)
    sigma2 = beta**2
    Sigma = sigma2 * np.eye(2)
    x = rng.multivariate_normal(mean=mu, cov=Sigma, size=N)
    x1, x2 = x[:, 0], x[:, 1]

    fig, ax = plt.subplots(figsize=(6, 5), dpi=180)
    xlim = (-4, 4)
    ylim = (-4, 4)
    cmap = plt.cm.viridis
    ax.set_facecolor(cmap(0.0))
    ax.hist2d(
        x1,
        x2,
        bins=80,
        range=[xlim, ylim],
        density=True,
        cmap=cmap,
        vmin=0.0,
    )
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")

    # Remove ticks (both marks and labels)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
    plt.tight_layout()
    plt.savefig(
        f"content/posts/generation_targets/distribution_2d_alpha{alpha:.2f}_beta{beta:.2f}.png",
        bbox_inches="tight",
    )
    plt.close()
