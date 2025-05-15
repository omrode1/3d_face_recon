import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
from utils.flame_model import FlameModel
import torch

# Paths
param_path = 'data/3dmm_fits/Front_params.json'  # Update if needed
image_path = 'data/processed/Front_processed.jpg'
model_path = 'models/flame2023/flame2023.pkl'

# Load FLAME model
flame = FlameModel(model_path, flame_version='2023')

# Load parameters
with open(param_path, 'r') as f:
    params = json.load(f)
scale = float(params['scale'])
tx = float(params['tx'])
ty = float(params['ty'])

# Apply numpy compatibility patches
if not hasattr(np, 'bool'):
    np.bool = bool
if not hasattr(np, 'object'):
    np.object = object
if not hasattr(np, 'str'):
    np.str = str
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'complex'):
    np.complex = complex
if not hasattr(np, 'unicode'):
    np.unicode = str

# Convert parameters to PyTorch tensors
shape_params = torch.tensor(params['shape_params'][0], dtype=torch.float32).unsqueeze(0)
exp_params = torch.tensor(params['exp_params'][0], dtype=torch.float32).unsqueeze(0)
pose_params = torch.tensor(params['pose_params'][0], dtype=torch.float32).unsqueeze(0)

# Get mesh vertices
vertices, _ = flame(shape_params, exp_params, pose_params)
vertices = vertices[0].detach().cpu().numpy()

# Load image
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
h, w = image.shape[:2]

# Project 3D vertices to 2D using camera parameters (match fitting logic)
projected = vertices[:, :2] * scale
projected += np.array([tx, ty])  # translation is in normalized space
projected[:, 0] = projected[:, 0] * w + w / 2
projected[:, 1] = projected[:, 1] * h + h / 2

print(f"Projected 2D range: min={projected.min(axis=0)}, max={projected.max(axis=0)}, mean={projected.mean(axis=0)}")

# Load MediaPipe-to-FLAME mapping
mapping_path = 'data/additional_resources/mediapipe_landmark_embedding.npz'
mapping_data = np.load(mapping_path, allow_pickle=True)
flame_lmk_indices = mapping_data['landmark_indices'][:68]

# Get model landmarks (indices used in fitting)
model_landmarks_2d = projected[flame_lmk_indices]

# Overlay only the FLAME mesh landmarks (red) and 2D detected landmarks (green)
overlay = image_rgb.copy()
for x, y in model_landmarks_2d.astype(int):
    if 0 <= x < w and 0 <= y < h:
        cv2.circle(overlay, (x, y), 2, (255, 0, 0), -1)  # Red for mesh landmarks

# Overlay 2D detected landmarks (green)
landmark_path = 'data/processed/Front_landmarks.json'  # Update if needed
try:
    with open(landmark_path, 'r') as f:
        landmarks = json.load(f)
    landmarks = np.array(landmarks)
    for x, y in landmarks.astype(int):
        cv2.circle(overlay, (x, y), 2, (0, 255, 0), -1)  # Green for detected landmarks
    print("\nFLAME mesh landmark coordinates (projected):\n", model_landmarks_2d)
    print("\n2D detected landmark coordinates:\n", landmarks)
except Exception as e:
    print(f"Could not overlay 2D landmarks: {e}")

plt.figure(figsize=(8, 8))
plt.imshow(overlay)
plt.title('Projected 3D Mesh Vertices on Image')
plt.axis('off')
plt.tight_layout()
plt.savefig('data/mesh_projection_overlay.png')
plt.show() 