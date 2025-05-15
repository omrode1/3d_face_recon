"""
Utility functions for loading 3D models including FLAME
Provides a wrapper to safely load FLAME model without chumpy dependency issues
"""

import os
import sys
import numpy as np
import torch
import pickle
from pathlib import Path

def load_flame_model(model_path):
    """
    Load FLAME model from pickle file, handling chumpy arrays
    
    This function loads the FLAME model pickle file and converts
    chumpy arrays to NumPy arrays to avoid dependency issues.
    
    Parameters:
    - model_path: path to FLAME model pickle file
    
    Returns:
    - model_data: dictionary with model data
    """
    try:
        # First check if the file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"FLAME model not found at {model_path}")
            
        # Try to load with different pickle versions and options
        try:
            # Try pickle5 first
            import pickle5
            with open(model_path, 'rb') as f:
                model_data = pickle5.load(f)
        except (ImportError, pickle.UnpicklingError):
            # Fall back to regular pickle with different encodings
            try:
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f, encoding='latin1')
            except:
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f, encoding='utf-8')
                
        # Convert chumpy arrays to numpy arrays
        # Identify keys that contain chumpy arrays
        for key in model_data.keys():
            if hasattr(model_data[key], 'r'):  # Chumpy arrays have a .r attribute
                try:
                    model_data[key] = np.array(model_data[key])
                except Exception as e:
                    print(f"Warning: Could not convert {key} to numpy array: {e}")
                    
            # Handle sparse matrices
            if hasattr(model_data[key], 'todense'):
                try:
                    model_data[key] = np.array(model_data[key].todense())
                except Exception as e:
                    print(f"Warning: Could not convert {key} to numpy array: {e}")
                    
            # Handle FLAME 2023 format
            if isinstance(model_data[key], np.ndarray):
                # Ensure arrays are in the correct format
                if model_data[key].dtype == np.object_:
                    try:
                        model_data[key] = np.array(model_data[key].tolist())
                    except Exception as e:
                        print(f"Warning: Could not convert {key} to numpy array: {e}")
                        
            # Handle boolean arrays
            if isinstance(model_data[key], np.ndarray) and model_data[key].dtype == bool:
                model_data[key] = model_data[key].astype(np.int8)
                
        return model_data, True
        
    except Exception as e:
        print(f"Error loading FLAME model: {e}")
        return None, False
        
def create_dummy_flame_model(n_vertices=5023, n_shape=100, n_exp=50):
    """
    Create a dummy FLAME model for testing or when the real model is unavailable
    
    Parameters:
    - n_vertices: number of vertices
    - n_shape: number of shape parameters
    - n_exp: number of expression parameters
    
    Returns:
    - model_data: dictionary with dummy model data
    """
    import trimesh
    
    # Create a simple sphere as template
    sphere = trimesh.creation.icosphere(subdivisions=4)
    vertices = sphere.vertices
    faces = sphere.faces
    
    # Resample to match vertex count if needed
    if len(vertices) != n_vertices:
        print(f"Note: Dummy model has {len(vertices)} vertices, target is {n_vertices}")
        # Use the smaller number
        n_vertices = min(len(vertices), n_vertices)
        vertices = vertices[:n_vertices]
        
    # Create model data dictionary
    model_data = {
        'v_template': vertices,
        'f': faces,
        'shapedirs': np.zeros((n_vertices, 3, n_shape)),
        'expressiondirs': np.zeros((n_vertices, 3, n_exp)),
        'J_regressor': np.zeros((15, n_vertices)),  # 15 joints
        'kintree_table': np.zeros((2, 15), dtype=np.int64),
        'weights': np.zeros((n_vertices, 15)),
        'posedirs': np.zeros((n_vertices, 3, 9)),
        'landmark_indices': list(range(18))
    }
    
    return model_data 