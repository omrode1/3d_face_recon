import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scripts.extract_landmarks import extract_landmarks

# Paths
uv_path = 'data/uv_coords.npy'
image_path = 'data/processed/Front_processed.jpg'  # Update if needed

# Load UV coordinates
uv_coords = np.load(uv_path)

# Print and inspect the range of UV coordinates
u_min, v_min = uv_coords.min(axis=0)
u_max, v_max = uv_coords.max(axis=0)
u_mean, v_mean = uv_coords.mean(axis=0)
print(f"U range: min={u_min:.4f}, max={u_max:.4f}, mean={u_mean:.4f}")
print(f"V range: min={v_min:.4f}, max={v_max:.4f}, mean={v_mean:.4f}")

# Count UVs outside [0, 1]
out_of_bounds = np.sum((uv_coords < 0) | (uv_coords > 1))
print(f"Number of UV coordinates outside [0, 1]: {out_of_bounds}")

# Scatter plot of UVs
plt.figure(figsize=(6, 6))
plt.scatter(uv_coords[:, 0], uv_coords[:, 1], s=1, alpha=0.7)
plt.title('UV Coordinates Scatter Plot')
plt.xlabel('U')
plt.ylabel('V')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.gca().invert_yaxis()
plt.grid(True)
plt.tight_layout()
plt.savefig('data/uv_scatter.png')
plt.show()

# Overlay UVs on input image
image = cv2.imread(image_path)
if image is None:
    print(f"Could not load image: {image_path}")
    exit(1)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
h, w = image.shape[:2]

# Convert UVs to pixel coordinates
uv_pixels = np.zeros_like(uv_coords)
uv_pixels[:, 0] = (uv_coords[:, 0] * w).clip(0, w-1)
uv_pixels[:, 1] = (uv_coords[:, 1] * h).clip(0, h-1)

# Draw UV points on the image
overlay = image_rgb.copy()
for x, y in uv_pixels.astype(int):
    cv2.circle(overlay, (x, y), 1, (255, 0, 0), -1)

plt.figure(figsize=(8, 8))
plt.imshow(overlay)
plt.title('UV Coordinates Overlay on Image')
plt.axis('off')
plt.tight_layout()
plt.savefig('data/uv_overlay.png')
plt.show()

# Visualize 2D landmarks detected by MediaPipe
landmarks_2d = extract_landmarks(image_path)
if landmarks_2d:
    image_lm = image_rgb.copy()
    for (x, y) in landmarks_2d:
        cv2.circle(image_lm, (int(x), int(y)), 2, (0, 255, 0), -1)
    plt.figure(figsize=(8, 8))
    plt.imshow(image_lm)
    plt.title('2D Landmarks Detected by MediaPipe')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('data/mediapipe_landmarks_overlay.png')
    plt.show()
else:
    print('No landmarks detected by MediaPipe.') 