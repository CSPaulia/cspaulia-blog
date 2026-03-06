import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# Visualize (in 2D) a Gaussian conditional path and its conditional vector field.
# Path:   P_t(·|z) = N(alpha_t z, sigma_t^2 I), with alpha_t=t, sigma_t=1-t.
# Field:  u_t(x|z) = (α̇_t - (σ̇_t/σ_t) α_t) z + (σ̇_t/σ_t) x.

z = np.array([3.0, 3.0], dtype=float)
frames = 10
fps = 2
eps = 1e-3  # avoid t=1 where sigma_t=0
out_path = "content/posts/generation_targets/conditional_vector_field_2d.gif"

# Make the Gaussian "smaller" (more concentrated): variance scale factor.
# Smaller -> tighter distribution. Try 0.3, 0.5, 1.0.
sigma_scale = 0.3

xlim = (-4.0, 4.0)
ylim = (-4.0, 4.0)
grid_step = 1.0
arrow_len = 0.35  # fixed arrow length (direction only)


def alpha(t: float) -> float:
	return t


def sigma(t: float) -> float:
	return sigma_scale * (1.0 - t)


def alpha_dot(_: float) -> float:
	return 1.0


def sigma_dot(_: float) -> float:
	return -sigma_scale


def u_field(t: float, x: np.ndarray) -> np.ndarray:
	a = alpha(t)
	s = max(sigma(t), eps)
	adot = alpha_dot(t)
	sdot = sigma_dot(t)
	ratio = sdot / s
	return (adot - ratio * a) * z[None, :] + ratio * x


def gaussian_density(t: float, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
	a = alpha(t)
	s = max(sigma(t), eps)
	mu = a * z
	dx = X - mu[0]
	dy = Y - mu[1]
	r2 = dx * dx + dy * dy
	coef = 1.0 / (2.0 * np.pi * s * s)
	return coef * np.exp(-0.5 * r2 / (s * s))


ts = np.linspace(0.0, 1.0 - eps, frames)

# background density grid (for imshow + contour)
nx = 220
ny = 220
xs = np.linspace(xlim[0], xlim[1], nx)
ys = np.linspace(ylim[0], ylim[1], ny)
X, Y = np.meshgrid(xs, ys)

# vector field grid (arrows at red-grid intersections)
gx = np.arange(xlim[0], xlim[1] + 1e-9, grid_step)
gy = np.arange(ylim[0], ylim[1] + 1e-9, grid_step)
GX, GY = np.meshgrid(gx, gy)
G0 = np.stack([GX.ravel(), GY.ravel()], axis=1)
grid_h, grid_w = GX.shape

fig, ax = plt.subplots(figsize=(5.2, 5.2), dpi=180)
ax.set_xlim(*xlim)
ax.set_ylim(*ylim)
ax.set_aspect("equal", adjustable="box")

# red grid (no tick labels) - will be animated by advecting the grid corners
ax.set_xticks(gx)
ax.set_yticks(gy)
ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)

# init density
d0 = gaussian_density(float(ts[0]), X, Y)
im = ax.imshow(
	np.log(d0 + 1e-12),
	extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
	origin="lower",
	cmap="viridis",
	interpolation="nearest",
)

# init contours: levels correspond to k-sigma circles (k independent of t)
k_levels = np.array([0.5, 1.0, 1.5, 2.0])
contour = None

# init moving grid points (Lagrangian): will be advected by dX/dt = u_t(X|z)
P = G0.copy()

# init vector field arrows (direction only)
U0 = u_field(float(ts[0]), P)
U0n = U0 / (np.linalg.norm(U0, axis=1, keepdims=True) + 1e-9)
q = ax.quiver(
	P[:, 0],
	P[:, 1],
	U0n[:, 0],
	U0n[:, 1],
	color="black",
	angles="xy",
	scale_units="xy",
	scale=1.0 / arrow_len,
	width=0.0032,
)

# init red grid lines (connect advected grid corners)
grid_lines = []
P2 = P.reshape(grid_h, grid_w, 2)
for r in range(grid_h):
	(line,) = ax.plot(P2[r, :, 0], P2[r, :, 1], color="red", alpha=0.35, linewidth=0.8)
	grid_lines.append(line)
for c in range(grid_w):
	(line,) = ax.plot(P2[:, c, 0], P2[:, c, 1], color="red", alpha=0.35, linewidth=0.8)
	grid_lines.append(line)

# mark z
ax.scatter([z[0]], [z[1]], s=35, color="crimson", zorder=5)


def update(frame_idx: int):
	global contour
	t = float(ts[frame_idx])

	# advect grid-corner points forward in time (Lagrangian view)
	if frame_idx > 0:
		t0 = float(ts[frame_idx - 1])
		dt = t - t0
		# midpoint (RK2) integration for stability
		k1 = u_field(t0, P)
		P_mid = P + 0.5 * dt * k1
		k2 = u_field(t0 + 0.5 * dt, P_mid)
		P[:] = P + dt * k2

	# update density image + contour
	d = gaussian_density(t, X, Y)
	im.set_data(np.log(d + 1e-12))

	if contour is not None:
		# Matplotlib compatibility: newer versions support QuadContourSet.remove()
		try:
			contour.remove()
		except Exception:
			for c in getattr(contour, "collections", []):
				c.remove()

	s = max(sigma(t), eps)
	peak = 1.0 / (2.0 * np.pi * s * s)
	levels = peak * np.exp(-0.5 * k_levels * k_levels)
	# Matplotlib requires contour levels to be strictly increasing.
	levels = np.unique(np.sort(levels))
	contour = ax.contour(X, Y, d, levels=levels, colors="white", linewidths=1.0, alpha=0.85)

	# update vector field arrows: both position (P) and direction u_t(P|z)
	U = u_field(t, P)
	Un = U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-9)
	q.set_offsets(P)
	q.set_UVC(Un[:, 0], Un[:, 1])

	# update red grid lines (connect the advected corners)
	P2 = P.reshape(grid_h, grid_w, 2)
	idx = 0
	for r in range(grid_h):
		grid_lines[idx].set_data(P2[r, :, 0], P2[r, :, 1])
		idx += 1
	for c in range(grid_w):
		grid_lines[idx].set_data(P2[:, c, 0], P2[:, c, 1])
		idx += 1

	ax.set_title(f"t={t:.2f}")
	return (im, q, *grid_lines)


images: list[Image.Image] = []

# Render frames manually to avoid PillowWriter/FuncAnimation swallowing the real exception
# (which can otherwise show up as "IndexError: list index out of range").
for k in range(frames):
	update(k)
	fig.canvas.draw()
	# Copy the buffer so each frame is independent (otherwise some viewers end up with only the last frame).
	rgba = np.asarray(fig.canvas.buffer_rgba()).copy()
	images.append(Image.fromarray(rgba, mode="RGBA"))

plt.close(fig)

if not images:
	raise RuntimeError("No frames rendered")

duration_ms = int(1000 / fps)
images[0].save(
	out_path,
	save_all=True,
	append_images=images[1:],
	duration=duration_ms,
	optimize=False,
	disposal=2,
	loop=0,
)

print(f"Saved: {out_path}")
