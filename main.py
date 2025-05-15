import os
import sys
import argparse
from pathlib import Path
import subprocess
import time

# Import the patch function
from utils.patch_chumpy import apply_patch
from scripts.convert_flame import convert_flame_model

def create_directories():
    """Create necessary directories for the pipeline"""
    directories = [
        "data/raw",
        "data/processed",
        "data/3dmm_fits",
        "data/obj",
        "models/flame",
        "models/flame2023"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True, parents=True)
        
def find_flame_model(model_dir):
    """
    Find the FLAME model in the specified directory.
    First tries FLAME 2023, then falls back to FLAME 2020.
    
    Returns:
    - model_path: path to the FLAME model
    - version: version of the FLAME model (2023 or 2020)
    """
    # Try FLAME 2023 first
    flame2023_path = Path("models/flame2023/flame2023.pkl")
    if flame2023_path.exists():
        return flame2023_path, "2023"
    
    # Fall back to FLAME 2020
    flame2020_path = Path(model_dir) / "FLAME2020.pkl"
    if flame2020_path.exists():
        return flame2020_path, "2020"
    
    # No model found
    return None, None

def download_flame_model(model_dir):
    """
    Instructions for downloading FLAME model
    
    Due to licensing restrictions, the FLAME model must be downloaded manually
    """
    # First check if FLAME 2023 is available
    flame_path, version = find_flame_model(model_dir)
    
    if flame_path is not None:
        print(f"Found FLAME {version} model at {flame_path}")
        return True
    
    # No model found, display instructions
    print("=" * 80)
    print("FLAME MODEL NOT FOUND")
    print("=" * 80)
    print("Please download the FLAME model from:")
    print("https://flame.is.tue.mpg.de/")
    print()
    print("After accepting the license agreement:")
    print("1. For FLAME 2023 (recommended):")
    print("   - Download FLAME 2023")
    print("   - Extract the downloaded zip file to models/flame2023/")
    print()
    print("2. Or for FLAME 2020:")
    print("   - Download FLAME 2020 (FLAME 2020 with 300 shape and 100 expression parameters)")
    print("   - Extract the downloaded zip file")
    print("   - Copy/move the FLAME2020.pkl file to:")
    print(f"   {Path(model_dir) / 'FLAME2020.pkl'}")
    print("=" * 80)
    return False

def run_pipeline(args):
    """Run the complete pipeline"""
    start_time = time.time()
    
    print("=" * 50)
    print("SINGLE-VIEW 3D FACE RECONSTRUCTION PIPELINE")
    print("=" * 50)
    
    # Apply the chumpy patch to fix NumPy compatibility
    print("\n[0/4] Applying compatibility patch...")
    patch_success = apply_patch()
    if not patch_success:
        print("Warning: Failed to apply compatibility patch. The pipeline may not work correctly.")
    
    # Check for FLAME model
    if not download_flame_model(args.model_dir):
        return
    
    # Get the FLAME model path
    flame_path, version = find_flame_model(args.model_dir)
    
    # Convert FLAME model to numpy format if using FLAME 2020
    if version == "2020":
        print("\n[1/4] Converting FLAME model to compatible format...")
        converted_flame_path = Path(args.model_dir) / "FLAME2020_numpy.npz"
        
        # Use original model as default in case conversion fails
        model_path = str(flame_path)
        
        # Only convert if not already converted
        if not converted_flame_path.exists():
            convert_success = convert_flame_model(flame_path, converted_flame_path)
            if convert_success and converted_flame_path.exists():
                print(f"Successfully converted FLAME model to {converted_flame_path}")
                model_path = str(converted_flame_path)
            else:
                print("Warning: Failed to convert FLAME model. Using original model.")
        else:
            print(f"Using previously converted FLAME model: {converted_flame_path}")
            model_path = str(converted_flame_path)
    else:
        # Using FLAME 2023, no conversion needed
        print(f"\n[1/4] Using FLAME 2023 model: {flame_path}")
        model_path = str(flame_path)
        
    # 1. Preprocess images
    print("\n[2/4] Preprocessing frontal images...")
    preprocess_cmd = [
        sys.executable, "scripts/preprocess.py",
        "--input", args.input_dir,
        "--output", args.processed_dir,
        "--device", args.device
    ]
    subprocess.run(preprocess_cmd)
    
    # 2. Fit 3DMM model directly to single image
    print("\n[3/4] Fitting 3D Morphable Model to frontal faces...")
    fit_cmd = [
        sys.executable, "scripts/fit_3dmm.py",
        "--input", args.processed_dir,
        "--output", args.fits_dir,
        "--model", model_path,
        "--device", args.device,
        "--iterations", str(args.iterations),
        "--flame-version", version,  # Add flame version parameter
        "--single-view", "true"  # Add flag for single-view processing
    ]
    subprocess.run(fit_cmd)
    
    # 3. Export 3D meshes
    print("\n[4/4] Exporting textured 3D meshes...")
    export_cmd = [
        sys.executable, "scripts/export_obj.py",
        "--input", args.fits_dir,
        "--images", args.processed_dir,
        "--output", args.obj_dir,
        "--model", model_path,
        "--device", args.device,
        "--flame-version", version  # Add flame version parameter
    ]
    subprocess.run(export_cmd)
    
    # Calculate total time
    total_time = time.time() - start_time
    print(f"\nTotal processing time: {total_time:.2f} seconds")
    
    # Print output location
    print("\nReconstruction complete!")
    print(f"3D meshes saved to: {args.obj_dir}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D Face Reconstruction Pipeline")
    
    # Directory arguments
    parser.add_argument("--input-dir", default="data/raw", help="Input directory with raw images")
    parser.add_argument("--processed-dir", default="data/processed", help="Directory for processed images")
    parser.add_argument("--fits-dir", default="data/3dmm_fits", help="Directory for 3DMM fits")
    parser.add_argument("--obj-dir", default="data/obj", help="Directory for output OBJ files")
    parser.add_argument("--model-dir", default="models/flame", help="Directory for 3D model files")
    
    # Processing parameters
    parser.add_argument("--device", default="cuda" if "cuda" in sys.modules else "cpu", 
                        help="Device to use (cuda/cpu)")
    parser.add_argument("--height", type=int, default=512, help="Image height")
    parser.add_argument("--width", type=int, default=512, help="Image width")
    parser.add_argument("--iterations", type=int, default=100, help="Optimization iterations")
    
    args = parser.parse_args()
    
    # Create necessary directories
    create_directories()
    
    # Run the pipeline
    run_pipeline(args) 