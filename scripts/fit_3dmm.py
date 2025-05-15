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
    np.unicode = str

import torch
import torch.nn as nn
import torch.optim as optim
import json
from pathlib import Path
import cv2
import os
import sys
import argparse
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.flame_model import FlameModel
from utils.renderer import Renderer

class FaceFitter:
    def __init__(self, flame_model_path, flame_version="2020", device='cuda' if torch.cuda.is_available() else 'cpu', single_view=False):
        """
        Initialize the face fitter with a FLAME model
        
        Parameters:
        - flame_model_path: path to FLAME model file
        - flame_version: version of the FLAME model (2020 or 2023)
        - device: device to use for computation
        - single_view: whether to use single-view approach (DECA-like)
        """
        print(f"Using device: {device}")
        print(f"Using {'single-view' if single_view else 'multi-view'} approach")
        
        # Initialize FLAME model
        self.flame = FlameModel(flame_model_path, flame_version=flame_version)
        self.flame.to(device)
        
        self.device = device
        self.single_view = single_view
        
        # Initialize renderer
        self.renderer = Renderer(device=device)
        
    def load_landmarks(self, landmark_path):
        """Load 2D landmarks from JSON file"""
        with open(landmark_path, 'r') as f:
            landmarks = json.load(f)
        return np.array(landmarks)
        
    def initialize_params(self):
        """Initialize FLAME parameters"""
        # Shape parameters (identity)
        shape_params = torch.zeros(1, self.flame.n_shape_params, device=self.device)
        
        # Expression parameters
        exp_params = torch.zeros(1, self.flame.n_exp_params, device=self.device)
        
        # Pose parameters (global rotation and jaw)
        pose_params = torch.zeros(1, self.flame.n_pose_params, device=self.device)
        
        # Initialize camera parameters
        scale = torch.tensor(4.0, requires_grad=True, device=self.device)
        tx = torch.tensor(0.0, requires_grad=True, device=self.device)
        ty = torch.tensor(0.0, requires_grad=True, device=self.device)
        tz = torch.tensor(0.0, requires_grad=True, device=self.device)
        
        # Make parameters require gradients
        shape_params.requires_grad = True
        exp_params.requires_grad = True
        pose_params.requires_grad = True
        scale.requires_grad = True
        tx.requires_grad = True
        ty.requires_grad = True
        tz.requires_grad = True
        
        return shape_params, exp_params, pose_params, scale, tx, ty, tz
        
    def project_vertices(self, vertices, cam_params, image_size):
        """Project 3D vertices to 2D using weak perspective projection"""
        # Extract camera parameters
        scale = cam_params[:, 0:1]
        rotation = cam_params[:, 1:4]
        
        # Apply rotation (simplified)
        # In a full implementation, convert rotation parameters to rotation matrix
        # For simplicity, we're just doing a basic projection here
        
        # Apply scale and perspective projection
        vertices_2d = scale * vertices[:, :, :2]
        
        # Convert to image coordinates (center of image)
        h, w = image_size
        vertices_2d[:, :, 0] = vertices_2d[:, :, 0] + w/2
        vertices_2d[:, :, 1] = vertices_2d[:, :, 1] + h/2
        
        return vertices_2d
        
    def landmark_loss(self, vertices_2d, landmarks_gt, flame_landmark_indices):
        """Compute landmark loss between projected vertices and ground truth landmarks"""
        # Check dimensions and adjust if needed
        # Our GT landmarks might have different count than the model's landmarks
        if flame_landmark_indices.shape[0] != landmarks_gt.shape[1]:
            print(f"Landmark count mismatch: model has {flame_landmark_indices.shape[0]}, GT has {landmarks_gt.shape[1]}")
            
            # If GT has more landmarks, use a subset
            if landmarks_gt.shape[1] > flame_landmark_indices.shape[0]:
                # Just use the first few landmarks
                landmarks_gt = landmarks_gt[:, :flame_landmark_indices.shape[0], :]
            else:
                # If model has more landmarks, select a subset from model
                # that matches GT count
                flame_landmark_indices = flame_landmark_indices[:landmarks_gt.shape[1]]
        
        # Select landmarks from vertices
        landmarks_pred = vertices_2d[:, flame_landmark_indices]
        
        # Compute MSE loss
        loss = torch.mean((landmarks_pred - landmarks_gt) ** 2)
        return loss
    
    def regularization_loss(self, shape_params, exp_params, pose_params):
        """Regularization to prevent extreme face shapes and poses"""
        shape_reg = torch.mean(shape_params ** 2)
        exp_reg = torch.mean(exp_params ** 2)
        pose_reg = torch.mean(pose_params[:, 3:] ** 2)  # Only regularize jaw pose
        
        return shape_reg + exp_reg + pose_reg
    
    def photometric_loss(self, rendered_image, target_image, mask=None):
        """
        Compute photometric loss between rendered and target images
        Used in single-view reconstruction to ensure texture consistency
        
        Parameters:
        - rendered_image: rendered face image tensor [B, H, W, 3]
        - target_image: target face image tensor [B, H, W, 3]
        - mask: optional mask to focus on face region [B, H, W, 1]
        
        Returns:
        - loss: photometric loss
        """
        if mask is not None:
            diff = (rendered_image - target_image) * mask
            loss = torch.sum(diff ** 2) / (torch.sum(mask) + 1e-6)
        else:
            diff = rendered_image - target_image
            loss = torch.mean(diff ** 2)
            
        return loss
    
    def fit_face(self, image_path, landmarks, iterations=100):
        """
        Fit the 3D face model to the input image and landmarks
        
        Parameters:
        - image_path: path to input image
        - landmarks: facial landmarks
        - iterations: number of optimization iterations
        
        Returns:
        - params: dictionary of optimized parameters
        """
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Failed to load image: {image_path}")
            return None
            
        image_height, image_width = image.shape[:2]
        
        # Prepare landmarks
        landmarks = np.array(landmarks)
        
        # Load MediaPipe-to-FLAME mapping
        mapping_path = 'data/additional_resources/mediapipe_landmark_embedding.npz'
        mapping_data = np.load(mapping_path, allow_pickle=True)
        flame_lmk_indices = mapping_data['landmark_indices'][:68]
        
        # Choose the appropriate fitting method based on single_view flag
        if self.single_view:
            return self.fit_face_single_view(image_path, landmarks, iterations)
        else:
            # Original multi-view method
            # Initialize parameters
            shape_params, exp_params, pose_params, scale, tx, ty, tz = self.initialize_params()
            
            # Optimizer
            optimizer = torch.optim.Adam([
                {'params': shape_params},
                {'params': exp_params},
                {'params': pose_params},
                {'params': scale, 'lr': 0.05},
                {'params': tx, 'lr': 0.05},
                {'params': ty, 'lr': 0.05},
                {'params': tz}
            ], lr=0.005)
            
            # Landmark loss weighting
            landmark_weights = torch.ones(landmarks.shape[0], device=self.device)
            # Emphasize eyes and mouth
            if landmarks.shape[0] >= 68:
                # Eyes: points 36-47
                landmark_weights[36:48] = 2.0
                # Mouth: points 48-67
                landmark_weights[48:68] = 2.0
            
            # Fitting loop
            for i in range(iterations):
                optimizer.zero_grad()
                
                # Forward pass
                vertices, _ = self.flame(shape_params, exp_params, pose_params)
                
                # Get model landmarks using mapped indices
                model_landmarks = vertices[:, flame_lmk_indices, :]
                
                # Scale model landmarks to image
                model_landmarks_2d = model_landmarks[:, :, :2]
                model_landmarks_2d = model_landmarks_2d * scale.view(-1, 1, 1)
                translation = torch.stack([tx, ty, tz]).view(1, 1, 3)
                model_landmarks_2d = model_landmarks_2d + translation[:, :, :2]

                # Convert to image space (keep as tensor)
                model_landmarks_2d = model_landmarks_2d.squeeze(0)
                model_landmarks_2d[:, 0] = model_landmarks_2d[:, 0] * image_width + image_width/2
                model_landmarks_2d[:, 1] = model_landmarks_2d[:, 1] * image_height + image_height/2

                # Ensure landmarks is a torch tensor on the correct device
                if not torch.is_tensor(landmarks):
                    landmarks_tensor = torch.tensor(landmarks, dtype=torch.float32, device=self.device)
                else:
                    landmarks_tensor = landmarks.to(self.device)

                # Check for landmark count mismatch
                if model_landmarks_2d.shape[0] != landmarks_tensor.shape[0]:
                    print(f"Landmark count mismatch: model has {model_landmarks_2d.shape[0]}, GT has {landmarks_tensor.shape[0]}")
                    min_landmarks = min(model_landmarks_2d.shape[0], landmarks_tensor.shape[0])
                    loss = torch.nn.functional.mse_loss(
                        model_landmarks_2d[:min_landmarks],
                        landmarks_tensor[:min_landmarks],
                        reduction='none'
                    )
                    loss = loss * landmark_weights[:min_landmarks].unsqueeze(1)
                else:
                    loss = torch.nn.functional.mse_loss(
                        model_landmarks_2d,
                        landmarks_tensor,
                        reduction='none'
                    )
                    loss = loss * landmark_weights.unsqueeze(1)
                
                # Add regularization for shape and expression parameters
                reg_shape = torch.mean(shape_params ** 2)
                reg_exp = torch.mean(exp_params ** 2)
                reg_pose = torch.mean(pose_params ** 2)
                
                # Total loss
                total_loss = torch.mean(loss) + 0.001 * reg_shape + 0.001 * reg_exp + 0.001 * reg_pose
                
                # Backward and optimize
                total_loss.backward()
                print(f"scale.grad: {scale.grad}, tx.grad: {tx.grad}, ty.grad: {ty.grad}, tz.grad: {tz.grad}")
                optimizer.step()
                
                # Print progress
                if (i+1) % 10 == 0:
                    print(f"Iteration {i+1}/{iterations}, Loss: {total_loss.item():.4f}")
            
            print(f"Optimized camera parameters: {scale.detach().cpu().numpy()}, {tx.detach().cpu().numpy()}, {ty.detach().cpu().numpy()}, {tz.detach().cpu().numpy()}")
            
            # Return optimized parameters
            params = {
                'shape_params': shape_params.detach().cpu().numpy().tolist(),
                'exp_params': exp_params.detach().cpu().numpy().tolist(),
                'pose_params': pose_params.detach().cpu().numpy().tolist(),
                'scale': scale.detach().cpu().numpy().tolist(),
                'tx': tx.detach().cpu().numpy().tolist(),
                'ty': ty.detach().cpu().numpy().tolist(),
                'tz': tz.detach().cpu().numpy().tolist()
            }
            
            return params
    
    def fit_face_single_view(self, image_path, landmarks, iterations=100):
        """
        DECA-like approach to fit the 3D face model to a single frontal image
        
        Parameters:
        - image_path: path to input image
        - landmarks: facial landmarks
        - iterations: number of optimization iterations
        
        Returns:
        - params: dictionary of optimized parameters
        """
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Failed to load image: {image_path}")
            return None
            
        image_height, image_width = image.shape[:2]
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert image to tensor
        image_tensor = torch.tensor(rgb_image, dtype=torch.float32).to(self.device) / 255.0
        if len(image_tensor.shape) == 3:  # Add batch dimension
            image_tensor = image_tensor.unsqueeze(0)
        
        # Prepare landmarks
        landmarks = torch.tensor(landmarks, dtype=torch.float32, device=self.device)
        if len(landmarks.shape) == 2:  # Add batch dimension
            landmarks = landmarks.unsqueeze(0)
        
        # Initialize parameters with stronger priors for single view
        shape_params, exp_params, pose_params, scale, tx, ty, tz = self.initialize_params()
        
        # Optimizer with different learning rates for different parameter groups
        optimizer = torch.optim.Adam([
            {'params': shape_params, 'lr': 0.001},  # Lower lr for shape
            {'params': exp_params, 'lr': 0.01},     # Higher lr for expressions
            {'params': pose_params, 'lr': 0.005},   # Medium lr for pose
            {'params': scale, 'lr': 0.01},           # Higher lr for scale
            {'params': tx, 'lr': 1.0},              # Higher lr for translation (pixel units)
            {'params': ty, 'lr': 1.0},              # Higher lr for translation (pixel units)
            {'params': tz, 'lr': 0.01}               # Higher lr for translation
        ])
        
        # Landmark loss weighting (focusing more on contour for single-view)
        landmark_weights = torch.ones(landmarks.shape[1], device=self.device)
        if landmarks.shape[1] >= 68:
            # Eyes: points 36-47
            landmark_weights[36:48] = 3.0  # Higher weight for eyes
            # Mouth: points 48-67
            landmark_weights[48:68] = 2.0  # Medium weight for mouth
            # Contour: points 0-16
            landmark_weights[0:17] = 4.0   # Highest weight for contour (important for 3D shape)
            # Nose: points 27-35
            landmark_weights[27:36] = 1.5  # Medium-low weight for nose
        
        # Create face mask (simplified)
        face_mask = torch.ones((image_height, image_width, 1), device=self.device)
        
        # DECA uses a coarse-to-fine approach
        # We'll implement a simplified version with two stages
        
        # Stage 1: Fit global shape, expression and pose (coarse)
        print("Stage 1: Fitting global face shape and pose...")
        for i in range(iterations // 2):  # First half of iterations
            optimizer.zero_grad()
            
            # Forward pass
            vertices, _ = self.flame(shape_params, exp_params, pose_params)
            
            # Get model landmarks using mapped indices
            model_landmarks = vertices[:, flame_lmk_indices, :]
            
            # Center the mesh landmarks
            center = model_landmarks.mean(dim=1, keepdim=True)
            model_landmarks_centered = model_landmarks - center
            
            # Project to 2D space
            model_landmarks_2d = model_landmarks_centered[:, :, :2]
            # Apply scale
            model_landmarks_2d = model_landmarks_2d * scale.view(-1, 1, 1)
            # Convert to image space
            model_landmarks_2d[:, :, 0] = model_landmarks_2d[:, :, 0] * image_width + image_width/2
            model_landmarks_2d[:, :, 1] = model_landmarks_2d[:, :, 1] * image_height + image_height/2
            # Apply translation (tx, ty) in pixel units
            model_landmarks_2d = model_landmarks_2d + torch.stack([tx, ty], dim=1).view(-1, 1, 2)
            
            # Compute landmark loss
            min_landmarks = min(model_landmarks_2d.shape[1], landmarks.shape[1])
            loss = torch.nn.functional.mse_loss(
                model_landmarks_2d[:, :min_landmarks, :],
                landmarks[:, :min_landmarks, :],
                reduction='none'
            )
            
            # Apply landmark weights
            weighted_loss = loss * landmark_weights[:min_landmarks].unsqueeze(0).unsqueeze(-1)
            landmark_loss = torch.mean(weighted_loss)
            
            # Stronger regularization for single view (prevents implausible shapes)
            shape_reg = torch.mean(shape_params ** 2) * 0.05  # Higher weight for shape regularization
            exp_reg = torch.mean(exp_params ** 2) * 0.05
            pose_reg = torch.mean(pose_params ** 2) * 0.1  # Higher weight for pose regularization
            
            # Symmetry constraint (optional - enforces some face symmetry)
            if shape_params.shape[1] >= 2:
                even_indices = torch.arange(0, shape_params.shape[1], 2, device=self.device)
                if even_indices.shape[0] > 1:
                    symmetry_loss = torch.mean(torch.abs(shape_params[:, even_indices[:-1]] - shape_params[:, even_indices[1:]]))
                else:
                    symmetry_loss = torch.tensor(0.0, device=self.device)
            else:
                symmetry_loss = torch.tensor(0.0, device=self.device)
            
            # Total loss
            total_loss = landmark_loss + shape_reg + exp_reg + pose_reg + symmetry_loss * 0.1
            
            # Backward and optimize
            total_loss.backward()
            optimizer.step()
            
            # Print progress
            if (i+1) % 5 == 0:
                print(f"Iteration {i+1}/{iterations//2}, Loss: {total_loss.item():.4f}")
        
        # Stage 2: Refine details (fine)
        print("Stage 2: Refining facial details...")
        # Reduce learning rates for fine tuning
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.5
            
        for i in range(iterations // 2, iterations):
            optimizer.zero_grad()
            
            # Forward pass
            vertices, _ = self.flame(shape_params, exp_params, pose_params)
            
            # Get model landmarks using mapped indices
            model_landmarks = vertices[:, flame_lmk_indices, :]
            
            # Center the mesh landmarks
            center = model_landmarks.mean(dim=1, keepdim=True)
            model_landmarks_centered = model_landmarks - center
            
            # Project to 2D space
            model_landmarks_2d = model_landmarks_centered[:, :, :2]
            # Apply scale
            model_landmarks_2d = model_landmarks_2d * scale.view(-1, 1, 1)
            # Convert to image space
            model_landmarks_2d[:, :, 0] = model_landmarks_2d[:, :, 0] * image_width + image_width/2
            model_landmarks_2d[:, :, 1] = model_landmarks_2d[:, :, 1] * image_height + image_height/2
            # Apply translation (tx, ty) in pixel units
            model_landmarks_2d = model_landmarks_2d + torch.stack([tx, ty], dim=1).view(-1, 1, 2)
            
            # Compute landmark loss - same as stage 1
            min_landmarks = min(model_landmarks_2d.shape[1], landmarks.shape[1])
            loss = torch.nn.functional.mse_loss(
                model_landmarks_2d[:, :min_landmarks, :],
                landmarks[:, :min_landmarks, :],
                reduction='none'
            )
            
            # Apply landmark weights - but with less influence than stage 1
            weighted_loss = loss * landmark_weights[:min_landmarks].unsqueeze(0).unsqueeze(-1) * 0.8
            landmark_loss = torch.mean(weighted_loss)
            
            # Lower regularization to allow for more detail fitting
            shape_reg = torch.mean(shape_params ** 2) * 0.01
            exp_reg = torch.mean(exp_params ** 2) * 0.02
            pose_reg = torch.mean(pose_params ** 2) * 0.05
            
            # Total loss (no symmetry constraint in stage 2 to allow for natural asymmetry)
            total_loss = landmark_loss + shape_reg + exp_reg + pose_reg
            
            # Backward and optimize
            total_loss.backward()
            optimizer.step()
            
            # Print progress
            if (i+1) % 5 == 0:
                print(f"Iteration {i+1}/{iterations}, Loss: {total_loss.item():.4f}")
        
        # Return optimized parameters
        params = {
            'shape_params': shape_params.detach().cpu().numpy().tolist(),
            'exp_params': exp_params.detach().cpu().numpy().tolist(),
            'pose_params': pose_params.detach().cpu().numpy().tolist(),
            'scale': scale.detach().cpu().numpy().tolist(),
            'tx': tx.detach().cpu().numpy().tolist(),
            'ty': ty.detach().cpu().numpy().tolist(),
            'tz': tz.detach().cpu().numpy().tolist()
        }
        
        return params
    
    def process_directory(self, input_dir="../data/processed", output_dir="../data/3dmm_fits"):
        """Process all preprocessed images in a directory"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # Find all processed images with landmarks
        processed_count = 0
        
        for img_path in input_path.glob("*_processed.jpg"):
            landmark_path = input_path / f"{img_path.stem.replace('_processed', '')}_landmarks.json"
            
            if landmark_path.exists():
                print(f"Fitting 3DMM to {img_path}...")
                try:
                    result = self.fit_face(img_path, self.load_landmarks(landmark_path))
                    if result is not None:
                        # Save parameters
                        output_file = output_path / f"{img_path.stem.replace('_processed', '')}_params.json"
                        with open(output_file, 'w') as f:
                            json.dump(result, f, indent=2)
                        processed_count += 1
                except Exception as e:
                    print(f"Error fitting {img_path}: {str(e)}")
            else:
                print(f"No landmarks found for {img_path}")
        
        print(f"Processed {processed_count} images")
        return processed_count

def process_images(args):
    """Process all images in the input directory"""
    # Initialize face fitter
    fitter = FaceFitter(args.model, flame_version=args.flame_version, device=args.device, 
                       single_view=args.single_view.lower() == "true")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Get all images in the input directory
    input_dir = Path(args.input)
    image_paths = list(input_dir.glob('*_processed.jpg')) + list(input_dir.glob('*_processed.jpeg')) + list(input_dir.glob('*_processed.png'))
    
    if len(image_paths) == 0:
        print(f"No processed images found in {args.input}")
        return
    
    # Process each image
    for image_path in image_paths:
        print(f"Fitting 3DMM to {image_path}...")
        
        # Load landmarks from accompanying file
        base_name = image_path.stem.replace('_processed', '')
        landmark_path = input_dir / f"{base_name}_landmarks.json"
        
        if not landmark_path.exists():
            print(f"No landmarks found for {image_path}, tried {landmark_path}")
            continue
            
        with open(landmark_path, 'r') as f:
            try:
                # Try different formats
                landmark_data = json.load(f)
                # Check if it's a list (direct coordinates) or a dictionary with 'landmarks' key
                if isinstance(landmark_data, list):
                    landmarks = np.array(landmark_data)
                elif isinstance(landmark_data, dict) and 'landmarks' in landmark_data:
                    landmarks = np.array(landmark_data['landmarks'])
                else:
                    print(f"Invalid landmark format in {landmark_path}")
                    continue
            except Exception as e:
                print(f"Error loading landmarks from {landmark_path}: {e}")
                continue
        
        # Fit the model
        params = fitter.fit_face(image_path, landmarks, iterations=args.iterations)
        
        if params is not None:
            # Save parameters
            output_path = output_dir / f"{base_name}_params.json"
            with open(output_path, 'w') as f:
                json.dump(params, f, indent=2)
    
    print(f"Processed {len(image_paths)} images")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit 3D Morphable Model to face images")
    parser.add_argument("--input", required=True, help="Input directory with preprocessed face images")
    parser.add_argument("--output", required=True, help="Output directory for 3DMM parameters")
    parser.add_argument("--model", required=True, help="Path to FLAME model file")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda/cpu)")
    parser.add_argument("--iterations", type=int, default=100, help="Number of optimization iterations")
    parser.add_argument("--flame-version", default="2020", choices=["2020", "2023"], help="FLAME model version")
    parser.add_argument("--single-view", default="false", choices=["true", "false"], help="Whether to use single-view approach")
    
    args = parser.parse_args()
    
    process_images(args)
