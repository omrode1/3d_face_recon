"""
Convert FLAME model to a simpler format without chumpy dependency

This script converts the FLAME model pickle file to a simpler numpy-based format
that can be loaded without chumpy dependency.
"""

import os
import sys
import numpy as np
import pickle
import argparse
from pathlib import Path
import json

def convert_flame_model(input_path, output_path):
    """
    Convert FLAME model from pickle to numpy-friendly format
    
    Parameters:
    - input_path: path to FLAME model pickle file
    - output_path: path to save the converted model
    
    Returns:
    - success: True if conversion was successful
    """
    try:
        print(f"Converting FLAME model from {input_path} to {output_path}")
        
        # Try to load the model
        try:
            # Try pickle5 first
            import pickle5
            with open(input_path, 'rb') as f:
                model_data = pickle5.load(f)
        except (ImportError, pickle.UnpicklingError):
            # Fall back to regular pickle
            with open(input_path, 'rb') as f:
                model_data = pickle.load(f, encoding='latin1')
                
        # Create a new model without chumpy dependencies
        converted_model = {}
                
        # Process each key in the original model
        for key, value in model_data.items():
            try:
                # Check if it's a chumpy array
                if hasattr(value, 'r'):
                    # Convert chumpy array to numpy array
                    converted_model[key] = np.array(value)
                else:
                    # Copy as is
                    converted_model[key] = value
            except Exception as e:
                print(f"Warning: Couldn't convert {key}: {e}")
                # Try a direct copy
                converted_model[key] = value
        
        # Save the converted model
        if output_path.endswith('.pkl') or output_path.endswith('.pickle'):
            # Save as pickle
            with open(output_path, 'wb') as f:
                pickle.dump(converted_model, f)
        elif output_path.endswith('.npz'):
            # Save as npz
            np.savez(output_path, **converted_model)
        else:
            # Default to npz
            np.savez(output_path, **converted_model)
            
        print(f"Successfully converted FLAME model to {output_path}")
        return True
        
    except Exception as e:
        print(f"Error converting FLAME model: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert FLAME model to numpy format")
    parser.add_argument("--input", required=True, help="Input FLAME model pickle file")
    parser.add_argument("--output", required=True, help="Output file path (.pkl or .npz)")
    
    args = parser.parse_args()
    
    convert_flame_model(args.input, args.output) 