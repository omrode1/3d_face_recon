"""
Patch for chumpy library to work with newer versions of NumPy
This patches the NumPy module to add the missing types that chumpy requires.
"""

import sys
import os
import importlib
import numpy as np

def apply_patch():
    """
    Apply the patch to fix chumpy compatibility with newer NumPy versions
    by monkey-patching NumPy module
    """
    try:
        # Check NumPy version
        numpy_version = np.__version__.split('.')
        major, minor = int(numpy_version[0]), int(numpy_version[1])
        
        # For older NumPy versions (pre-1.20), these types exist directly 
        # and we don't need to patch anything
        if major == 1 and minor < 20:
            print(f"Using NumPy {np.__version__}, no patching needed")
            # Still try to import chumpy to check if it works
            import chumpy
            return True
            
        # For newer NumPy versions, add the missing attributes 
        # only if they don't exist
        patched = False
        
        if not hasattr(np, 'bool'):
            np.bool = bool
            patched = True
            
        if not hasattr(np, 'int'):
            np.int = int
            patched = True
            
        if not hasattr(np, 'float'):
            np.float = float
            patched = True
            
        if not hasattr(np, 'complex'):
            np.complex = complex
            patched = True
            
        if not hasattr(np, 'object'):
            np.object = object
            patched = True
            
        if not hasattr(np, 'str'):
            np.str = str
            patched = True
            
        if not hasattr(np, 'unicode'):
            np.unicode = str  # In Python 3, unicode is just str
            patched = True
            
        # Test if chumpy can now import correctly
        import chumpy
        
        if patched:
            print(f"Successfully patched NumPy {np.__version__} for chumpy compatibility")
        else:
            print(f"NumPy {np.__version__} already compatible with chumpy")
            
        return True
    
    except Exception as e:
        print(f"Error patching NumPy for chumpy: {str(e)}")
        return False

if __name__ == "__main__":
    apply_patch() 