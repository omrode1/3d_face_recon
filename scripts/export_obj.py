import numpy as np
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
    np.unicode = str  # In Python 3, str is unicode





import json
import os
import sys
import argparse
from pathlib import Path
import cv2
import torch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.flame_model import FlameModel
from utils.renderer import Renderer

class MeshExporter:
    def __init__(self, flame_model_path, flame_version="2020", device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the mesh exporter
        
        Parameters:
        - flame_model_path: path to FLAME model file
        - flame_version: version of the FLAME model (2020 or 2023)
        - device: 'cuda' or 'cpu'
        """
        self.device = device
        print(f"Using device: {device}")
        
        # Initialize FLAME model
        self.flame = FlameModel(flame_model_path, flame_version=flame_version)
        self.flame.to(device)
        
        # Initialize renderer
        self.renderer = Renderer(device=device)
        
    def load_params(self, param_path):
        """Load parameters from JSON file"""
        with open(param_path, 'r') as f:
            params = json.load(f)
            
        # Convert parameters to numpy arrays, handling nested lists
        shape_params = np.array(params['shape_params'][0], dtype=np.float32)
        exp_params = np.array(params['exp_params'][0], dtype=np.float32)
        pose_params = np.array(params['pose_params'][0], dtype=np.float32)
        cam_params = np.array(params['cam_params'][0], dtype=np.float32)
        
        # Debug: Print parameter statistics
        print("\n3DMM Parameter Statistics:")
        print(f"Shape parameters (mean face deviation):")
        print(f"  Mean: {np.mean(shape_params):.4f}")
        print(f"  Std: {np.std(shape_params):.4f}")
        print(f"  Max deviation: {np.max(np.abs(shape_params)):.4f}")
        
        print(f"\nExpression parameters:")
        print(f"  Mean: {np.mean(exp_params):.4f}")
        print(f"  Std: {np.std(exp_params):.4f}")
        print(f"  Max deviation: {np.max(np.abs(exp_params)):.4f}")
        
        print(f"\nPose parameters (in degrees):")
        print(f"  Yaw: {np.degrees(pose_params[0]):.2f}°")
        print(f"  Pitch: {np.degrees(pose_params[1]):.2f}°")
        print(f"  Roll: {np.degrees(pose_params[2]):.2f}°")
        
        print(f"\nCamera parameters:")
        print(f"  Scale: {cam_params[0]:.4f}")
        print(f"  Translation: ({cam_params[1]:.4f}, {cam_params[2]:.4f})")
        
        return shape_params, exp_params, pose_params, cam_params
        
    def generate_uv_map(self, vertices, image_path, output_path):
        """
        Generate UV map and texture for the mesh
        
        Parameters:
        - vertices: 3D vertices of the mesh
        - image_path: path to input image
        - output_path: path to save the texture
        
        Returns:
        - uv_coords: UV coordinates
        - texture: texture image
        """
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Failed to load image: {image_path}")
            return None, None
            
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create a better UV mapping based on 3D face geometry
        # Calculate bounding box of vertices
        min_x, max_x = np.min(vertices[:, 0]), np.max(vertices[:, 0])
        min_y, max_y = np.min(vertices[:, 1]), np.max(vertices[:, 1])
        min_z, max_z = np.min(vertices[:, 2]), np.max(vertices[:, 2])
        
        # Create a visibility mask for single-view reconstruction
        # This helps us identify which parts of the face are visible in the frontal view
        # Vertices with negative Z (facing the camera) are more visible
        z_normalized = (vertices[:, 2] - min_z) / (max_z - min_z + 1e-10)
        visibility = np.clip(1.0 - z_normalized, 0.0, 1.0) ** 2  # Square to emphasize frontal parts
        
        # Normalize based on the bounding box
        normalized_vertices = vertices.copy()
        
        # Normalize x and y coordinates to [0, 1] range
        # We invert Y for texture coords (V coordinate needs to be flipped)
        normalized_vertices[:, 0] = (vertices[:, 0] - min_x) / (max_x - min_x + 1e-10)  # U coordinate
        normalized_vertices[:, 1] = 1.0 - (vertices[:, 1] - min_y) / (max_y - min_y + 1e-10)  # V coordinate (flipped)
        
        # Improved UV mapping strategy for single-view reconstruction
        # Use cylindrical projection for better mapping around the face
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        center_z = (min_z + max_z) / 2
        
        # Calculate cylindrical coordinates
        rel_x = vertices[:, 0] - center_x
        rel_y = vertices[:, 1] - center_y
        rel_z = vertices[:, 2] - center_z
        
        # Cylindrical projection with fading based on face orientation
        # This gives better stretching behavior for side face regions
        radius = np.sqrt(rel_x**2 + rel_z**2)
        theta = np.arctan2(rel_z, rel_x)  # -π to π
        
        # Create UV coordinates with cylindrical projection for horizontal wrapping
        uv_coords = np.zeros((vertices.shape[0], 2))
        
        # U coordinate: use angle for horizontal wrapping
        uv_coords[:, 0] = (theta + np.pi) / (2 * np.pi)  # Map from -π,π to 0,1
        
        # V coordinate: normalize height with adjustment for better face region focus
        height_normalized = (vertices[:, 1] - min_y) / (max_y - min_y + 1e-10)
        uv_coords[:, 1] = 1.0 - height_normalized  # Invert for UV texture coordinates
        
        # Adjust UV coordinates to focus on the face region
        # Front-facing vertices (negative Z) get priority in the texture space
        # This is particularly helpful for single-view cases where we only see frontal face
        front_factor = np.minimum(1.0, np.maximum(0.0, -vertices[:, 2] / (abs(min_z) + 1e-10)))
        
        # Move frontal vertices more toward the center of the texture
        center_offset_u = uv_coords[:, 0] - 0.5
        center_offset_v = uv_coords[:, 1] - 0.5
        
        # Reduce the offset for frontal vertices, keeping them closer to center
        uv_coords[:, 0] = 0.5 + center_offset_u * (1.0 - 0.2 * front_factor)
        uv_coords[:, 1] = 0.5 + center_offset_v * (1.0 - 0.2 * front_factor)
        
        # Create texture from image
        h, w = image_rgb.shape[:2]
        
        # Create a new texture image with appropriate size
        texture_size = max(h, w, 1024)  # Use at least 1024x1024
        texture = np.ones((texture_size, texture_size, 3), dtype=np.uint8) * 200  # Light gray background
        
        # Calculate scaling factors
        scale_x = 0.8 * texture_size / w  # Leave a margin
        scale_y = 0.8 * texture_size / h  # Leave a margin
        
        # Calculate offsets to center the image
        offset_x = (texture_size - w * scale_x) // 2
        offset_y = (texture_size - h * scale_y) // 2
        
        # Resize the image to fit the texture
        resized_image = cv2.resize(image_rgb, (int(w * scale_x), int(h * scale_y)))
        
        # Place the resized image in the texture
        texture[
            int(offset_y):int(offset_y + h * scale_y),
            int(offset_x):int(offset_x + w * scale_x)
        ] = resized_image
        
        # Scale UV coordinates to match the texture layout
        # Use a smaller scale factor to keep UVs more in the center where image is
        uv_coords[:, 0] = uv_coords[:, 0] * 0.7 + 0.15  # Scale and offset U
        uv_coords[:, 1] = uv_coords[:, 1] * 0.7 + 0.15  # Scale and offset V
        
        # Save texture
        cv2.imwrite(str(output_path), cv2.cvtColor(texture, cv2.COLOR_RGB2BGR))
        
        return uv_coords, texture
        
    def export_obj(self, param_file, image_path, output_dir):
        """
        Export 3D face mesh as OBJ with texture
        
        Parameters:
        - param_file: path to FLAME parameters
        - image_path: path to input image for texture
        - output_dir: directory to save the output
        
        Returns:
        - output_path: path to exported OBJ file
        """
        # Load parameters
        shape_params, exp_params, pose_params, cam_params = self.load_params(param_file)
        
        # Run FLAME model
        vertices, _ = self.flame(shape_params, exp_params, pose_params)
        
        # Convert to numpy
        vertices_np = vertices[0].detach().cpu().numpy()
        faces_np = self.flame.faces
        
        # Prepare output paths
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        obj_path = output_path / f"{Path(param_file).stem.replace('_params', '')}.obj"
        texture_path = output_path / f"{Path(param_file).stem.replace('_params', '')}_texture.png"
        
        # Generate UV coordinates and texture
        uv_coords, texture = self.generate_uv_map(vertices_np, image_path, texture_path)
        
        if uv_coords is None or texture is None:
            print(f"Failed to generate UV map for {param_file}")
            return None
            
        # Apply symmetry to make the complete face texture more realistic
        # This is helpful for single-view cases where only the frontal face is visible
        # We'll write a specific method in the renderer to handle this
        # Export mesh with texture
        self.renderer.export_mesh(vertices_np, faces_np, str(obj_path), texture, uv_coords)
        print(f"Exported mesh to {obj_path}")
        
        return str(obj_path)
        
    def process_directory(self, input_dir, image_dir, output_dir):
        """
        Process all parameter files in a directory
        
        Parameters:
        - input_dir: directory containing parameter files
        - image_dir: directory containing images
        - output_dir: directory to save output OBJ files
        """
        input_path = Path(input_dir)
        image_path = Path(image_dir)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # Find all parameter files
        param_files = list(input_path.glob("*_params.json"))
        
        if len(param_files) == 0:
            print(f"No parameter files found in {input_dir}")
            return
            
        for param_file in param_files:
            # Find corresponding image
            image_base = param_file.stem.replace("_params", "")
            image_file = image_path / f"{image_base}_processed.jpg"
            
            if not image_file.exists():
                # Try without _processed suffix
                image_file = image_path / f"{image_base}.jpg"
                
            if not image_file.exists():
                # Try other extensions
                for ext in ['.jpeg', '.png']:
                    image_file = image_path / f"{image_base}{ext}"
                    if image_file.exists():
                        break
                        
            if not image_file.exists():
                # Try processed extensions
                for ext in ['.jpg', '.jpeg', '.png']:
                    image_file = image_path / f"{image_base}_processed{ext}"
                    if image_file.exists():
                        break
                        
            if not image_file.exists():
                print(f"No image found for {param_file}")
                continue
                
            # Export OBJ
            try:
                self.export_obj(param_file, image_file, output_dir)
            except Exception as e:
                print(f"Error exporting mesh for {param_file}: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export 3D face meshes as OBJ files")
    parser.add_argument("--input", default="../data/3dmm_fits", help="Input directory with parameter files")
    parser.add_argument("--images", default="../data/processed", help="Directory with input images")
    parser.add_argument("--output", default="../data/obj", help="Output directory for OBJ files")
    parser.add_argument("--model", default="../models/flame/FLAME2020.pkl", help="Path to FLAME model file")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda/cpu)")
    parser.add_argument("--flame-version", default="2020", choices=["2020", "2023"], help="FLAME model version")
    
    args = parser.parse_args()
    
    exporter = MeshExporter(args.model, flame_version=args.flame_version, device=args.device)
    exporter.process_directory(args.input, args.images, args.output) 