import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

N = 100  # grid size
dx = 1.0

# Position grids
x = np.linspace(-3, 3, N)  # Centered at 0 for symmetric tanh
y = np.linspace(-3, 3, N)
X, Y = np.meshgrid(x, y)

# Smooth transition zone using tanh
Du = 0.01 + 0.01 * (1 + np.tanh(X)) / 2  # Range: [0.01, 0.02]
Dv = Du / 2  # Or you can define differently if needed


# Standard Gray-Scott parameters for spot pattern
F = 0.028
k = 0.064

# Compute safe dt based on max Du (von Neumann stability)
Du_max = Du.max()
dt = dx**2 / (4 * Du_max) * 0.95  # add safety factor

U = np.ones((N, N))
V = np.zeros((N, N))

# Add noise
U += 0.05 * np.random.rand(N, N)
V += 0.05 * np.random.rand(N, N)

# Seed a central square of higher V and lower U
r = 10
U[N//2 - r:N//2 + r, N//2 - r:N//2 + r] = 0.5
V[N//2 - r:N//2 + r, N//2 - r:N//2 + r] = 0.25


def laplacian(Z):
    return (
        -4 * Z
        + np.roll(Z, 1, axis=0) + np.roll(Z, -1, axis=0)
        + np.roll(Z, 1, axis=1) + np.roll(Z, -1, axis=1)
    ) / dx**2


def update(frame, img, U, V, Du, Dv, F, k, dx, dt):
    Lu = laplacian(U)
    Lv = laplacian(V)
    uvv = U * V * V

    # Update U and V — DO NOT add combined_derivative!
    U += (Du * Lu - uvv + F * (1 - U)) * dt
    V += (Dv * Lv + uvv - (F + k) * V) * dt

    # Clip to avoid numerical overflow
    U = np.clip(U, 0, 1)
    V = np.clip(V, 0, 1)

    img.set_data(V)
    return [img]


fig, ax = plt.subplots(figsize=(6, 6))
img = ax.imshow(V, cmap='plasma', interpolation='bilinear', vmin=0, vmax=1)
ax.set_title("Gray-Scott V field")
cbar = fig.colorbar(img, ax=ax)
cbar.set_label('[V]')

ani = animation.FuncAnimation(
    fig, update, fargs=(img, U, V, Du, Dv, F, k, dx, dt),
    frames=2000, interval=30, blit=True
)

plt.show()
