import matplotlib.pyplot as plt
import numpy as np

F = np.linspace(0.020, 0.050, 6)
k = np.linspace(0.040, 0.070, 6)

fig, ax = plt.subplots(figsize=(10, 10))
extent1 = 0.005
gap = 0.001


# Plot each image
for i, f_val in enumerate(F):
    for j, k_val in enumerate(k):
        imp_path = f"/LF_F{f_val:.3f}_k{k_val:.3f}.png"
        try:
            img = plt.imread(imp_path)
            ax.imshow(img, extent=[f_val - extent1, f_val + extent1,
                                   k_val - extent1, k_val + extent1], aspect='auto')
        except FileNotFoundError:
            print(f"Image not found: {imp_path}")

# Add dummy scalar mappable for colorbar
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

# Adjust these limits based on what your grayscale data actually ranged between
vmin = 0.0
vmax = 0.5
cmap = "viridis"  # Make sure it's the one you used when saving the images

sm = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
sm.set_array([])  # Required for older versions of matplotlib

# Add the colorbar
cbar = fig.colorbar(sm, ax=ax, orientation='vertical')
cbar.set_label("Concentration (e.g., V)")

# Axes settings
ax.set_xlim(0.020 - 2.5 * gap, F[-1] + gap)
ax.set_ylim(0.040 - 2.5 * gap, k[-1] + gap)
ax.set_xlabel('F (U Generation Rate)')
ax.set_ylabel('k (V Kill Rate)')
ax.set_xticks(F)
ax.set_yticks(k)
plt.title('Phase Diagram of Gray-Scott Model Zoomed In')
plt.show()
