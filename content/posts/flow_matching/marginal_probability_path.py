from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


NUM_SAMPLES = 150_000
TIMES = np.linspace(0.0, 1.0, 6)
LIMITS = (-4.0, 4.0)
SQUARE_HALF_WIDTH = 0.55
OUTPUT_PATH = Path(__file__).with_name("marginal_probability_path.png")

rng = np.random.default_rng(0)

# A 4 x 4 checkerboard data distribution, following the marginal-path
# illustration from the lecture.
coordinates = np.array([-2.4, -0.8, 0.8, 2.4])
square_centers = np.array(
    [
        (x, y)
        for row, y in enumerate(coordinates)
        for column, x in enumerate(coordinates)
        if (row + column) % 2 == 0
    ]
)

selected_squares = rng.integers(0, len(square_centers), size=NUM_SAMPLES)
z = square_centers[selected_squares] + rng.uniform(
    -SQUARE_HALF_WIDTH,
    SQUARE_HALF_WIDTH,
    size=(NUM_SAMPLES, 2),
)
epsilon = rng.normal(size=(NUM_SAMPLES, 2))

figure, axes = plt.subplots(
    1,
    len(TIMES),
    figsize=(24, 5),
    dpi=180,
    facecolor="black",
)
cmap = plt.cm.viridis

for axis, time in zip(axes, TIMES):
    samples = time * z + (1.0 - time) * epsilon

    axis.set_facecolor(cmap(0.0))
    axis.hist2d(
        samples[:, 0],
        samples[:, 1],
        bins=100,
        range=[LIMITS, LIMITS],
        density=True,
        cmap=cmap,
        vmin=0.0,
        vmax=0.18,
    )
    axis.set_xlim(*LIMITS)
    axis.set_ylim(*LIMITS)
    axis.set_aspect("equal", adjustable="box")
    axis.set_title(rf"$t = {time:.2f}$", color="white", fontsize=16, pad=10)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.tick_params(
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )
    for spine in axis.spines.values():
        spine.set_color("white")
        spine.set_linewidth(2)

figure.subplots_adjust(left=0.01, right=0.99, bottom=0.04, top=0.88, wspace=0.08)
figure.savefig(
    OUTPUT_PATH,
    facecolor=figure.get_facecolor(),
    bbox_inches="tight",
    pad_inches=0.05,
)
plt.close(figure)
