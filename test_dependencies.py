import sys
import importlib
import subprocess
import os
from pathlib import Path

def check_package(package_name):
    """Check if a package is installed"""
    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        return False

def test_dependencies():
    """Test if all required dependencies are installed"""
    required_packages = [
        'numpy',
        'opencv-python',
        'torch',
        'torchvision',
        'scipy',
        'matplotlib',
        'trimesh',
        'PIL',
        'face_alignment',
        'dlib'
    ]
    
    missing_packages = []
    
    print("Testing dependencies...")
    print("-" * 50)
    
    for package in required_packages:
        # Handle special cases
        if package == 'opencv-python':
            package_to_test = 'cv2'
        elif package == 'PIL':
            package_to_test = 'PIL.Image'
        else:
            package_to_test = package
            
        if check_package(package_to_test):
            print(f"✅ {package} is installed")
        else:
            print(f"❌ {package} is NOT installed")
            missing_packages.append(package)
    
    print("-" * 50)
    
    if missing_packages:
        print(f"Missing packages: {', '.join(missing_packages)}")
        print("Please install missing packages with:")
        print(f"pip install {' '.join(missing_packages)}")
    else:
        print("All required packages are installed! 🎉")
    
    return len(missing_packages) == 0

def test_pytorch_cuda():
    """Test if PyTorch CUDA is working"""
    if not check_package('torch'):
        print("PyTorch is not installed, skipping CUDA test")
        return False
        
    import torch
    
    print("\nTesting PyTorch CUDA availability...")
    print("-" * 50)
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        cuda_count = torch.cuda.device_count()
        print(f"CUDA device count: {cuda_count}")
        
        for i in range(cuda_count):
            print(f"CUDA Device {i}: {torch.cuda.get_device_name(i)}")
        
        print("PyTorch CUDA test: Creating tensor on GPU...")
        try:
            x = torch.tensor([1.0, 2.0, 3.0], device='cuda')
            print(f"✅ Successfully created tensor on GPU: {x}")
        except Exception as e:
            print(f"❌ Failed to create tensor on GPU: {str(e)}")
            return False
            
        return True
    else:
        print("CUDA is not available. Using CPU.")
        return False

def test_face_alignment():
    """Test face_alignment package with a sample image"""
    if not (check_package('face_alignment') and check_package('cv2')):
        print("face_alignment or OpenCV is not installed, skipping face detection test")
        return False
        
    print("\nTesting face_alignment...")
    print("-" * 50)
    
    # Check for sample image
    sample_dir = Path("data/sample")
    sample_dir.mkdir(exist_ok=True, parents=True)
    
    sample_image = sample_dir / "sample.jpg"
    
    if not sample_image.exists():
        print("No sample image found. Creating a simple test image...")
        
        # Create a simple image with a rectangle representing a face
        import cv2
        import numpy as np
        
        img = np.ones((300, 300, 3), dtype=np.uint8) * 255
        cv2.rectangle(img, (100, 100), (200, 200), (0, 0, 0), 2)
        cv2.circle(img, (130, 130), 5, (0, 0, 0), -1)  # Left eye
        cv2.circle(img, (170, 130), 5, (0, 0, 0), -1)  # Right eye
        cv2.line(img, (130, 170), (170, 170), (0, 0, 0), 2)  # Mouth
        
        cv2.imwrite(str(sample_image), img)
        print(f"Created sample image at {sample_image}")
    
    try:
        import face_alignment
        import numpy as np
        import torch
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device=device)
        
        # Load image
        import cv2
        image = cv2.imread(str(sample_image))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect landmarks
        landmarks = fa.get_landmarks(image_rgb)
        
        if landmarks is None:
            print("❌ No face detected in the sample image")
            return False
        else:
            print(f"✅ Detected {len(landmarks)} face(s) with landmarks")
            
            # Draw landmarks on image
            for lm in landmarks:
                for (x, y) in lm:
                    cv2.circle(image, (int(x), int(y)), 1, (0, 255, 0), -1)
            
            # Save result
            result_path = sample_dir / "sample_landmarks.jpg"
            cv2.imwrite(str(result_path), image)
            print(f"Saved landmark visualization to {result_path}")
            
            return True
    except Exception as e:
        print(f"❌ Error testing face_alignment: {str(e)}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("3D FACE RECONSTRUCTION DEPENDENCY TEST")
    print("=" * 60)
    
    # Test dependencies
    deps_ok = test_dependencies()
    
    # Test PyTorch CUDA
    cuda_ok = test_pytorch_cuda()
    
    # Test face_alignment
    face_ok = test_face_alignment()
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Required packages: {'✅ All installed' if deps_ok else '❌ Some missing'}")
    print(f"PyTorch CUDA: {'✅ Working' if cuda_ok else '❌ Not available/working'}")
    print(f"Face alignment: {'✅ Working' if face_ok else '❌ Not working'}")
    
    if deps_ok and face_ok:
        print("\n✅ Your system is ready to run the 3D face reconstruction pipeline!")
        print(f"GPU acceleration: {'✅ Available' if cuda_ok else '❌ Not available (CPU mode)'}")
    else:
        print("\n❌ Some components are not working correctly.")
        print("Please fix the issues above before running the pipeline.")
    
if __name__ == "__main__":
    main() 