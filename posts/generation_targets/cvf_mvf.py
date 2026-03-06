import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

mean = np.array([0.0, 0.0])
cov = np.array([
    [5.0, 0.0],
    [0.0, 5.0]
])

samples = np.random.multivariate_normal(mean=mean, cov=cov, size=1000)

mean_data = np.array([
    [-10.0, -10.0],
    [10.0, 10.0],
    [-10.0, 10.0],
    [10.0, -10.0]
])
cov_data = np.array([
    [[0.5, 0.0], [0.0, 0.5]],
    [[0.5, 0.0], [0.0, 0.5]],
    [[0.5, 0.0], [0.0, 0.5]],
    [[0.5, 0.0], [0.0, 0.5]]
])

samples_data = []
for mean, cov in zip(mean_data, cov_data):
    samples_data.append(np.random.multivariate_normal(mean=mean, cov=cov, size=250))

samples_data = np.concatenate(samples_data, axis=0)


def alpha_t(t):
    return t


def sigma_t(t):
    return 1.0 - t


def d_alpha_t(t):
    return 1.0


def d_sigma_t(t):
    return -1.0


def conditional_probability_mean(z, t):
    """Mean of P_t(.|z) = N(alpha_t z, sigma_t^2 I)."""
    return alpha_t(t) * z


def conditional_probability_cov(t, dim=2):
    """Covariance of P_t(.|z) = N(alpha_t z, sigma_t^2 I)."""
    sigma = sigma_t(t)
    return (sigma ** 2) * np.eye(dim)


def sample_conditional_probability(z, t, num_samples=1):
    """Sample x ~ P_t(.|z)."""
    z = np.asarray(z)
    dim = z.shape[-1]
    mean = conditional_probability_mean(z, t)
    cov = conditional_probability_cov(t, dim=dim)
    return np.random.multivariate_normal(mean=mean, cov=cov, size=num_samples)


def sample_marginal_probability(z_samples, t, num_samples=None):
    """Sample x ~ P_t by first sampling z ~ p_data, then x ~ P_t(.|z)."""
    z_samples = np.asarray(z_samples)

    if num_samples is None or num_samples >= len(z_samples):
        chosen_z = z_samples
    else:
        indices = np.random.choice(len(z_samples), size=num_samples, replace=False)
        chosen_z = z_samples[indices]

    noise = np.random.normal(size=chosen_z.shape)
    return alpha_t(t) * chosen_z + sigma_t(t) * noise


def conditional_vector_field(x, z, t, eps=1e-6):
    """
    Conditional vector field
    u_t(x|z) = (d alpha_t - (d sigma_t / sigma_t) alpha_t) z + (d sigma_t / sigma_t) x.
    For alpha_t=t, sigma_t=1-t, it simplifies to (z - x) / (1 - t).
    """
    x = np.asarray(x)
    z = np.asarray(z)
    sigma = max(sigma_t(t), eps)
    coeff_z = d_alpha_t(t) - (d_sigma_t(t) / sigma) * alpha_t(t)
    coeff_x = d_sigma_t(t) / sigma
    return coeff_z * z + coeff_x * x


def gaussian_density(x, mean, cov):
    """Multivariate Gaussian density."""
    x = np.asarray(x)
    mean = np.asarray(mean)
    dim = x.shape[-1]
    diff = x - mean
    inv_cov = np.linalg.inv(cov)
    det_cov = np.linalg.det(cov)
    norm = np.sqrt(((2.0 * np.pi) ** dim) * det_cov)
    exponent = -0.5 * diff @ inv_cov @ diff
    return np.exp(exponent) / norm


def conditional_density(x, z, t, eps=1e-6):
    """Density P_t(x|z) for the Gaussian conditional path."""
    z = np.asarray(z)
    dim = z.shape[-1]
    sigma = max(sigma_t(t), eps)
    mean = alpha_t(t) * z
    cov = (sigma ** 2) * np.eye(dim)
    return gaussian_density(x, mean, cov)


def marginal_vector_field(x, z_samples, t, eps=1e-12):
    """
    Monte-Carlo approximation of the marginal vector field:

        u_t(x) = \int u_t(x|z) p_data(z|x) dz
               = \frac{\int u_t(x|z) p_t(x|z) p_data(z) dz}{p_t(x)}

    Using empirical samples z_i ~ p_data, we approximate:

        u_t(x) \approx \frac{\sum_i u_t(x|z_i) p_t(x|z_i)}{\sum_i p_t(x|z_i)}.
    """
    x = np.asarray(x)
    z_samples = np.asarray(z_samples)

    weights = np.array([conditional_density(x, z, t) for z in z_samples])
    weights_sum = np.sum(weights)

    cvf_values = np.array([conditional_vector_field(x, z, t) for z in z_samples])
    normalized_weights = weights / weights_sum
    return np.sum(cvf_values * normalized_weights[:, None], axis=0)


def make_grid(xmin=-3.5, xmax=3.5, ymin=-3.5, ymax=3.5, num=15):
    x = np.linspace(xmin, xmax, num)
    y = np.linspace(ymin, ymax, num)
    xx, yy = np.meshgrid(x, y)
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1)
    return xx, yy, grid


def infer_plot_bounds(points, padding=2.0):
    min_xy = np.min(points, axis=0) - padding
    max_xy = np.max(points, axis=0) + padding
    return min_xy[0], max_xy[0], min_xy[1], max_xy[1]


def add_gaussian_ellipse(ax, mean, cov, n_std=2.0, edgecolor="black", linestyle="-", linewidth=1.5, alpha=0.9):
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width, height = 2.0 * n_std * np.sqrt(eigenvalues)

    ellipse = Ellipse(
        xy=mean,
        width=width,
        height=height,
        angle=angle,
        fill=False,
        edgecolor=edgecolor,
        linestyle=linestyle,
        linewidth=linewidth,
        alpha=alpha,
    )
    ax.add_patch(ellipse)


def add_distribution_references(ax, mean=mean, cov=cov):
    add_gaussian_ellipse(ax, mean, cov, n_std=2.0, edgecolor="tab:green", linestyle="-", linewidth=1.8)
    ax.scatter([mean[0]], [mean[1]], color="tab:green", s=40, marker="o", label="p0 Gaussian")

    for idx, (component_mean, component_cov) in enumerate(zip(mean_data, cov_data)):
        add_gaussian_ellipse(ax, component_mean, component_cov, n_std=2.0, edgecolor="tab:red", linestyle="--", linewidth=1.5)
        label = "pdata Gaussian components" if idx == 0 else None
        ax.scatter([component_mean[0]], [component_mean[1]], color="tab:red", s=35, marker="x", label=label)


def plot_distributions(ax):
    ax.scatter(samples[:, 0], samples[:, 1], s=10, alpha=0.2, label="p0 ~ N(0, I)")
    ax.scatter(samples_data[:, 0], samples_data[:, 1], s=10, alpha=0.2, label="pdata")
    ax.set_title("Initial and data distributions")
    ax.set_aspect("equal")
    ax.legend()


def plot_conditional_path(ax, z, t_values, xmin, xmax, ymin, ymax):
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(t_values)))
    for t, color in zip(t_values, colors):
        x_t = sample_conditional_probability(z, t, num_samples=150)
        ax.scatter(x_t[:, 0], x_t[:, 1], s=12, alpha=0.25, color=color, label=f"t={t:.2f}")

    add_distribution_references(ax, mean=conditional_probability_mean(z, t_values[0]), cov=conditional_probability_cov(t_values[0]))
    ax.scatter([z[0]], [z[1]], color="red", s=80, marker="x", label="target z")
    ax.set_title("Conditional probability path")
    ax.set_aspect("equal")
    ax.legend(fontsize=8)


def plot_marginal_path(ax, z_samples, t_values, xmin, xmax, ymin, ymax, num_samples=400):
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(t_values)))
    for t, color in zip(t_values, colors):
        x_t = sample_marginal_probability(z_samples, t, num_samples=num_samples)
        ax.scatter(x_t[:, 0], x_t[:, 1], s=10, alpha=0.2, color=color, label=f"t={t:.2f}")

    add_distribution_references(ax, mean=conditional_probability_mean(z_samples[0], t_values[0]), cov=conditional_probability_cov(t_values[0]))
    ax.set_title("Marginal probability path")
    ax.set_aspect("equal")
    ax.legend(fontsize=8)


def plot_vector_field(ax, xx, yy, u, v, background_points=None, highlight_point=None, title=""):
    if background_points is not None:
        ax.scatter(background_points[:, 0], background_points[:, 1], s=8, alpha=0.12, color="gray")

    ax.quiver(xx, yy, u.reshape(xx.shape), v.reshape(yy.shape), color="tab:blue", alpha=0.85)

    if highlight_point is not None:
        ax.scatter([highlight_point[0]], [highlight_point[1]], color="red", s=80, marker="x")

    ax.set_title(title)
    ax.set_aspect("equal")


def visualize_process(save_path="content/posts/generation_targets/cvf_mvf_visualization.png"):
    selected_z = samples_data[0]
    xmin, xmax, ymin, ymax = infer_plot_bounds(np.vstack([samples, samples_data]))
    t_values = [0.0, 0.25, 0.5, 0.75, 1.0]

    fig, axes = plt.subplots(1, 2, figsize=(17, 5))

    # plot_distributions(axes[0])
    plot_conditional_path(axes[0], selected_z, t_values=t_values, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
    plot_marginal_path(axes[1], samples_data, t_values=t_values, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)

    for ax in axes:
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    t = 0.5
    z = samples_data[0]
    x = sample_conditional_probability(z, t, num_samples=1)[0]

    cvf = conditional_vector_field(x, z, t)
    mvf = marginal_vector_field(x, samples_data, t)

    print("z:", z)
    print("x ~ P_t(.|z):", x)
    print("conditional vector field u_t(x|z):", cvf)
    print("marginal vector field u_t(x):", mvf)

    visualize_process()

