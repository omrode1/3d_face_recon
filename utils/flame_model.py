import torch
import torch.nn as nn
import numpy as np
import pickle
from pathlib import Path
import sys
import os
from utils.model_loader import load_flame_model, create_dummy_flame_model
import scipy.sparse

class FlameModel(nn.Module):
    """
    FLAME (Faces Learned with an Articulated Model and Expressions) face model
    
    This is a PyTorch implementation wrapper for the FLAME model
    https://flame.is.tue.mpg.de/
    """
    def __init__(self, model_path, flame_version="2023", n_shape=100, n_exp=50):
        super(FlameModel, self).__init__()
        
        # Check if model path exists
        if not Path(model_path).exists():
            raise ValueError(f"FLAME model path does not exist: {model_path}")
        
        # Set model version
        self.flame_version = flame_version
        print(f"Loading FLAME {flame_version} model")
        
        # Try to load the model
        try:
            # Check if it's an NPZ file
            if model_path.endswith('.npz'):
                print(f"Loading NPZ FLAME model from {model_path}")
                model_data = dict(np.load(model_path, allow_pickle=True))
                success = True
            else:
                # Load based on FLAME version
                if flame_version == "2023":
                    # FLAME 2023 uses a different format without chumpy dependencies
                    with open(model_path, 'rb') as f:
                        model_data = pickle.load(f, encoding='latin1')
                    success = True
                else:
                    # Try using the model loader for FLAME 2020 PKL files
                    model_data, success = load_flame_model(model_path)
        except Exception as e:
            print(f"Error loading FLAME model: {e}")
            success = False
            model_data = None
        
        if not success or model_data is None:
            print("Creating a simplified dummy model instead")
            model_data = create_dummy_flame_model(n_shape=n_shape, n_exp=n_exp)
        else:
            print("Successfully loaded FLAME model")
        
        # FLAME 2023 has a slightly different key structure
        if flame_version == "2023":
            # Map keys to common format
            if 'v_template' not in model_data and 'shapedirs' in model_data:
                if hasattr(model_data['shapedirs'], 'shape'):
                    num_vertices = model_data['shapedirs'].shape[0] // 3
                    model_data['v_template'] = np.zeros((num_vertices, 3))
            
            # Handle different face/faces key
            if 'f' not in model_data and 'faces' in model_data:
                model_data['f'] = model_data['faces']
            
            # Handle different shape/expression key names
            if 'shapedirs' not in model_data and 'shape_basis' in model_data:
                model_data['shapedirs'] = model_data['shape_basis']
            
            if 'expressiondirs' not in model_data and 'expression_basis' in model_data:
                model_data['expressiondirs'] = model_data['expression_basis']
            
        # Extract necessary components
        self.v_template = torch.tensor(np.array(model_data['v_template']), dtype=torch.float32)
        self.faces = np.array(model_data['f'] if 'f' in model_data else model_data.get('faces', []))
        
        # Shape and expression basis - convert if they are chumpy arrays
        if 'shapedirs' in model_data:
            shapedirs_np = np.array(model_data['shapedirs'])
            if len(shapedirs_np.shape) == 3:
                self.shapedirs = torch.tensor(shapedirs_np[:, :, :n_shape], dtype=torch.float32)
            elif len(shapedirs_np.shape) == 2:
                # Reshape for FLAME 2023 format
                n_vertices = shapedirs_np.shape[0] // 3
                self.shapedirs = torch.tensor(
                    shapedirs_np.reshape(-1, n_shape).reshape(n_vertices, 3, n_shape),
                    dtype=torch.float32
                )
            else:
                # Fallback
                print(f"Warning: Unexpected shapedirs shape: {shapedirs_np.shape}")
                self.shapedirs = torch.zeros((self.v_template.shape[0], 3, n_shape), dtype=torch.float32)
        else:
            self.shapedirs = torch.zeros((self.v_template.shape[0], 3, n_shape), dtype=torch.float32)
        
        if 'expressiondirs' in model_data:
            expressiondirs_np = np.array(model_data['expressiondirs'])
            if len(expressiondirs_np.shape) == 3:
                self.expressiondirs = torch.tensor(expressiondirs_np[:, :, :n_exp], dtype=torch.float32)
            elif len(expressiondirs_np.shape) == 2:
                # Reshape for FLAME 2023 format
                n_vertices = expressiondirs_np.shape[0] // 3
                self.expressiondirs = torch.tensor(
                    expressiondirs_np.reshape(-1, n_exp).reshape(n_vertices, 3, n_exp),
                    dtype=torch.float32
                )
            else:
                # Fallback
                print(f"Warning: Unexpected expressiondirs shape: {expressiondirs_np.shape}")
                self.expressiondirs = torch.zeros((self.v_template.shape[0], 3, n_exp), dtype=torch.float32)
        else:
            self.expressiondirs = torch.zeros((self.v_template.shape[0], 3, n_exp), dtype=torch.float32)
        
        # Pose related
        if 'J_regressor' in model_data:
            # Handle numpy.object_ arrays by converting sparse matrices to dense
            try:
                jr = model_data['J_regressor']
                # Check if it's a sparse matrix directly
                if isinstance(jr, scipy.sparse.csr_matrix) or isinstance(jr, scipy.sparse.csc_matrix):
                    # Convert sparse matrix to dense
                    jr_dense = jr.todense()
                    self.J_regressor = torch.tensor(jr_dense, dtype=torch.float32)
                # Check if it's a numpy.object_ array (array of sparse matrices)
                elif isinstance(jr, np.ndarray) and jr.dtype == np.object_:
                    # Get the number of joints (rows in regressor)
                    n_joints = len(jr)
                    # Create dense regressor matrix
                    jr_dense = np.zeros((n_joints, self.v_template.shape[0]), dtype=np.float32)
                    
                    # Convert each sparse matrix to dense
                    for i in range(n_joints):
                        if jr[i] is not None:
                            # Check if it's a sparse matrix
                            if hasattr(jr[i], 'todense'):
                                jr_dense[i] = jr[i].todense()
                            # Handle any other format
                            else:
                                jr_dense[i] = np.array(jr[i])
                    
                    self.J_regressor = torch.tensor(jr_dense, dtype=torch.float32)
                else:
                    # Regular dense array
                    self.J_regressor = torch.tensor(np.array(jr), dtype=torch.float32)
            except Exception as e:
                print(f"Warning: Error processing J_regressor: {e}")
                # Create a dummy regressor if conversion fails
                self.J_regressor = torch.zeros((15, self.v_template.shape[0]), dtype=torch.float32)
        else:
            # Create a dummy regressor
            self.J_regressor = torch.zeros((15, self.v_template.shape[0]), dtype=torch.float32)
        
        if 'kintree_table' in model_data:
            try:
                self.kintree_table = np.array(model_data['kintree_table'], dtype=np.int64)
            except:
                # Create a dummy kintree_table
                print("Warning: Could not convert kintree_table")
                self.kintree_table = np.zeros((2, 15), dtype=np.int64)
        else:
            # Create a dummy kintree_table
            self.kintree_table = np.zeros((2, 15), dtype=np.int64)
        
        if 'weights' in model_data:
            try:
                self.weights = torch.tensor(np.array(model_data['weights']), dtype=torch.float32)
            except:
                # Create dummy weights
                print("Warning: Could not convert weights")
                self.weights = torch.zeros((self.v_template.shape[0], 15), dtype=torch.float32)
        else:
            # Create dummy weights
            self.weights = torch.zeros((self.v_template.shape[0], 15), dtype=torch.float32)
        
        # LBS related
        if 'posedirs' in model_data:
            try:
                posedirs_np = np.array(model_data['posedirs'])
                self.posedirs = torch.tensor(posedirs_np, dtype=torch.float32)
            except:
                # Create dummy posedirs
                print("Warning: Could not convert posedirs")
                self.posedirs = torch.zeros((self.v_template.shape[0], 3, 9), dtype=torch.float32)
        else:
            # Create dummy posedirs
            self.posedirs = torch.zeros((self.v_template.shape[0], 3, 9), dtype=torch.float32)
        
        # Landmark indices for face alignment
        if 'landmark_indices' in model_data:
            try:
                landmark_indices = model_data['landmark_indices']
                if isinstance(landmark_indices, np.ndarray) and landmark_indices.dtype == np.object_:
                    # Handle special case from NPZ file
                    landmark_indices = list(range(68))
                self.landmark_indices = torch.tensor(landmark_indices, dtype=torch.long)
            except:
                # Create dummy landmark indices - use 68 to match typical facial landmarks
                print("Warning: Could not convert landmark_indices")
                self.landmark_indices = torch.tensor(list(range(68)), dtype=torch.long)
        else:
            # Create dummy landmark indices
            self.landmark_indices = torch.tensor(list(range(68)), dtype=torch.long)
            
        # Move all tensors to the same device
        self.to('cpu')  # Start on CPU, can be moved to GPU later
        
        # Model settings
        self.n_shape_params = n_shape
        self.n_exp_params = n_exp
        self.n_pose_params = 6  # 3 for global rotation, 3 for jaw rotation
        
    def forward(self, shape_params, exp_params, pose_params):
        """
        Forward pass of FLAME model
        
        Parameters:
        - shape_params: shape parameters, tensor of shape [batch_size, n_shape_params]
        - exp_params: expression parameters, tensor of shape [batch_size, n_exp_params]
        - pose_params: pose parameters, tensor of shape [batch_size, n_pose_params]
            - first 3 parameters for global rotation (axis-angle)
            - last 3 parameters for jaw rotation (axis-angle)
            
        Returns:
        - vertices: tensor of shape [batch_size, n_vertices, 3]
        - joints: tensor of shape [batch_size, n_joints, 3]
        """
        batch_size = shape_params.shape[0]
        device = shape_params.device
        
        # Move template to device if needed
        v_template = self.v_template.to(device).unsqueeze(0).expand(batch_size, -1, -1)
        
        # Reshape shape parameters if needed
        if len(shape_params.shape) < 3:
            # Add a dimension if shape_params is [batch_size, n_shape_params]
            shape_params = shape_params.unsqueeze(2)
        
        if len(exp_params.shape) < 3:
            # Add a dimension if exp_params is [batch_size, n_exp_params]
            exp_params = exp_params.unsqueeze(2)
        
        # Apply shape and expression blendshapes
        # Simple blend instead of einsum when using a simplified model
        try:
            v_shaped = v_template + torch.einsum('bik,jkl->bjl', [shape_params, self.shapedirs.to(device)])
            v_shaped_express = v_shaped + torch.einsum('bik,jkl->bjl', [exp_params, self.expressiondirs.to(device)])
        except RuntimeError:
            # Alternative simpler blending for shape and expression
            print("Using simpler blending method")
            v_shaped = v_template
            v_shaped_express = v_template
        
        # Just return the shaped vertices without posing
        vertices = v_shaped_express
        
        # Compute joints (simplified)
        try:
            joints = torch.einsum('ijk,kl->ijl', [vertices, self.J_regressor.to(device)])
        except RuntimeError:
            # Simple fallback for joints
            joints = torch.zeros((batch_size, 15, 3), device=device)
        
        return vertices, joints
        
    def to(self, device):
        """Move model to device"""
        super(FlameModel, self).to(device)
        self.v_template = self.v_template.to(device)
        self.shapedirs = self.shapedirs.to(device)
        self.expressiondirs = self.expressiondirs.to(device)
        self.J_regressor = self.J_regressor.to(device)
        self.weights = self.weights.to(device)
        self.posedirs = self.posedirs.to(device)
        self.landmark_indices = self.landmark_indices.to(device)
        return self 