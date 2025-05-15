import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import cv2
from utils.flame_model import FlameModel

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
if not hasattr(np, 'unicode'):
    np.unicode = str
if not hasattr(np, 'complex'):
    np.complex = complex

def load_mediapipe_embedding(embedding_path):
    """
    Load the MediaPipe landmark embedding.
    
    Parameters:
    - embedding_path: Path to the mediapipe_landmark_embedding.npz file
    
    Returns:
    - landmark_indices: Array of landmark indices
    - lmk_face_idx: Array of face indices for landmarks
    - lmk_b_coords: Barycentric coordinates for landmarks
    """
    data = np.load(embedding_path, allow_pickle=True)
    return data['landmark_indices'], data['lmk_face_idx'], data['lmk_b_coords']

def map_landmarks_to_flame(landmarks_2d, flame_model, image_path, embedding_path):
    """
    Map 2D facial landmarks to the FLAME mesh and compute UV mapping using MediaPipe embedding.
    
    Parameters:
    - landmarks_2d: List of (x, y) coordinates of facial landmarks
    - flame_model: FLAME model instance
    - image_path: Path to the input image
    - embedding_path: Path to the mediapipe_landmark_embedding.npz file
    
    Returns:
    - uv_coords: UV coordinates for the FLAME mesh
    """
    # Load the MediaPipe landmark embedding
    landmark_indices, lmk_face_idx, lmk_b_coords = load_mediapipe_embedding(embedding_path)
    
    # Load the image to get its dimensions
    image = cv2.imread(image_path)
    image_height, image_width = image.shape[:2]
    
    # Initialize UV coordinates array
    uv_coords = np.zeros((flame_model.v_template.shape[0], 2))
    
    # Get face vertices
    faces = flame_model.faces
    
    # Map landmarks using the embedding
    for i, (x, y) in enumerate(landmarks_2d):
        if i < len(landmark_indices):
            # Get the face index and barycentric coordinates for this landmark
            face_idx = lmk_face_idx[i]
            b_coords = lmk_b_coords[i]
            
            # Get the vertices of the face
            face_vertices = faces[face_idx]
            
            # Compute weighted UV coordinates using barycentric coordinates
            for j, vertex_idx in enumerate(face_vertices):
                uv_coords[vertex_idx] += np.array([x / image_width, y / image_height]) * b_coords[j]
    
    return uv_coords

if __name__ == "__main__":
    # Example usage
    from scripts.extract_landmarks import extract_landmarks
    from utils.flame_model import FlameModel
    
    # Load the FLAME model
    flame_model = FlameModel("models/flame2023/flame2023.pkl", flame_version="2023")
    
    # Extract landmarks from the image
    image_path = "data/processed/Front_processed.jpg"  # Update with your image path
    landmarks_2d = extract_landmarks(image_path)
    
    if landmarks_2d:
        # Path to the MediaPipe landmark embedding
        embedding_path = "data/additional_resources/mediapipe_landmark_embedding.npz"
        
        # Map landmarks to FLAME mesh
        uv_coords = map_landmarks_to_flame(landmarks_2d, flame_model, image_path, embedding_path)
        print(f"Computed UV coordinates for {len(uv_coords)} vertices.")
        
        # Save the UV coordinates for later use
        output_path = "data/uv_coords.npy"
        np.save(output_path, uv_coords)
        print(f"Saved UV coordinates to {output_path}")
    else:
        print("No landmarks found.") 