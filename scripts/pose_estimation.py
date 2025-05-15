import cv2
import numpy as np
import json
from pathlib import Path
import argparse
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class CameraPoseEstimator:
    """
    Estimate relative camera poses from facial landmarks
    Uses PnP and essential matrix decomposition for multiview setup
    """
    def __init__(self):
        # 3D reference points for standard face model
        # These are approximate 3D positions of standard facial landmarks
        # In a real implementation, these would come from a 3D face model
        # For simplicity, we're using a basic set of points
        self.face_3d_model = np.array([
            # Nose tip
            [0.0, 0.0, 0.0],
            # Left eye outer corner
            [-25.0, 30.0, -10.0],
            # Right eye outer corner
            [25.0, 30.0, -10.0],
            # Left mouth corner
            [-20.0, -30.0, -10.0],
            # Right mouth corner
            [20.0, -30.0, -10.0],
            # Chin
            [0.0, -55.0, -15.0],
            # Forehead
            [0.0, 60.0, -15.0]
        ])
        
        # Indices of the corresponding landmarks in the detected face landmarks
        # These indices depend on the landmark detection method
        # For face_alignment (68 landmarks), these might be:
        self.landmark_indices = {
            'nose_tip': 30,      # Nose tip
            'left_eye': 36,      # Left eye corner
            'right_eye': 45,     # Right eye corner
            'left_mouth': 48,    # Left mouth corner
            'right_mouth': 54,   # Right mouth corner
            'chin': 8,           # Chin
            'forehead': 27       # Forehead (approximation)
        }
        
        # Camera matrix (will be computed based on image size)
        self.camera_matrix = None
        self.dist_coeffs = np.zeros((4, 1))  # Assume no lens distortion
        
    def set_camera_matrix(self, image_size):
        """Set camera intrinsic parameters based on image size"""
        fx = image_size[1]  # Focal length x (width)
        fy = image_size[0]  # Focal length y (height)
        cx = image_size[1] / 2  # Principal point x
        cy = image_size[0] / 2  # Principal point y
        
        self.camera_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float64)
        
    def get_face_3d_points(self, landmarks):
        """
        Extract 3D points from landmarks corresponding to the 3D model points
        """
        points_3d = self.face_3d_model
        points_2d = np.array([
            landmarks[self.landmark_indices['nose_tip']],
            landmarks[self.landmark_indices['left_eye']],
            landmarks[self.landmark_indices['right_eye']],
            landmarks[self.landmark_indices['left_mouth']],
            landmarks[self.landmark_indices['right_mouth']],
            landmarks[self.landmark_indices['chin']],
            landmarks[self.landmark_indices['forehead']]
        ], dtype=np.float64)
        
        return points_3d, points_2d
        
    def estimate_pose(self, landmarks, image_size):
        """
        Estimate camera pose from facial landmarks using PnP
        
        Parameters:
        - landmarks: 2D facial landmarks, numpy array of shape [n_landmarks, 2]
        - image_size: tuple of (height, width)
        
        Returns:
        - rotation: rotation matrix
        - translation: translation vector
        - success: whether pose estimation was successful
        """
        # Set camera matrix if not already set
        if self.camera_matrix is None:
            self.set_camera_matrix(image_size)
            
        # Get corresponding 3D and 2D points
        points_3d, points_2d = self.get_face_3d_points(landmarks)
        
        # Solve PnP
        success, rotation_vec, translation_vec = cv2.solvePnP(
            points_3d, points_2d, self.camera_matrix, self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        # Convert rotation vector to matrix
        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        
        return rotation_mat, translation_vec, success
        
    def estimate_relative_pose(self, landmarks1, landmarks2, image_size):
        """
        Estimate relative camera pose between two images
        
        Parameters:
        - landmarks1: 2D facial landmarks from first image
        - landmarks2: 2D facial landmarks from second image
        - image_size: tuple of (height, width)
        
        Returns:
        - R: rotation matrix between cameras
        - t: translation vector between cameras
        - success: whether pose estimation was successful
        """
        # Set camera matrix if not already set
        if self.camera_matrix is None:
            self.set_camera_matrix(image_size)
            
        # Find common points between the two images
        # For simplicity, we use the same set of points as for PnP
        points1 = np.array([
            landmarks1[self.landmark_indices['nose_tip']],
            landmarks1[self.landmark_indices['left_eye']],
            landmarks1[self.landmark_indices['right_eye']],
            landmarks1[self.landmark_indices['left_mouth']],
            landmarks1[self.landmark_indices['right_mouth']],
            landmarks1[self.landmark_indices['chin']],
            landmarks1[self.landmark_indices['forehead']]
        ], dtype=np.float64)
        
        points2 = np.array([
            landmarks2[self.landmark_indices['nose_tip']],
            landmarks2[self.landmark_indices['left_eye']],
            landmarks2[self.landmark_indices['right_eye']],
            landmarks2[self.landmark_indices['left_mouth']],
            landmarks2[self.landmark_indices['right_mouth']],
            landmarks2[self.landmark_indices['chin']],
            landmarks2[self.landmark_indices['forehead']]
        ], dtype=np.float64)
        
        # Compute essential matrix
        E, mask = cv2.findEssentialMat(points1, points2, self.camera_matrix)
        
        # Recover rotation and translation from essential matrix
        _, R, t, _ = cv2.recoverPose(E, points1, points2, self.camera_matrix)
        
        return R, t, mask.sum() > 5  # Success if more than 5 inliers
        
    def process_image_pair(self, landmark_path1, landmark_path2, output_dir, image_size=None):
        """
        Process a pair of images and estimate relative pose
        
        Parameters:
        - landmark_path1: path to landmarks of first image
        - landmark_path2: path to landmarks of second image
        - output_dir: directory to save pose information
        - image_size: optional image size, tuple of (height, width)
        
        Returns:
        - R: rotation matrix between cameras
        - t: translation vector between cameras
        """
        # Load landmarks
        with open(landmark_path1, 'r') as f:
            landmarks1 = np.array(json.load(f))
            
        with open(landmark_path2, 'r') as f:
            landmarks2 = np.array(json.load(f))
            
        # Use provided image size or default
        if image_size is None:
            image_size = (512, 512)  # Default size
            
        # Estimate relative pose
        R, t, success = self.estimate_relative_pose(landmarks1, landmarks2, image_size)
        
        if success:
            # Save pose information
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True, parents=True)
            
            pose_file = output_path / f"{Path(landmark_path1).stem}_to_{Path(landmark_path2).stem}_pose.json"
            pose_data = {
                'rotation': R.tolist(),
                'translation': t.flatten().tolist(),
                'source': str(landmark_path1),
                'target': str(landmark_path2)
            }
            
            with open(pose_file, 'w') as f:
                json.dump(pose_data, f)
                
            print(f"Saved pose information to {pose_file}")
        else:
            print(f"Failed to estimate pose between {landmark_path1} and {landmark_path2}")
            
        return R, t
        
    def process_multiview(self, input_dir, output_dir, image_size=None):
        """
        Process multiple views and estimate relative poses
        
        Parameters:
        - input_dir: directory containing landmark files
        - output_dir: directory to save pose information
        - image_size: optional image size, tuple of (height, width)
        """
        # Get all landmark files
        input_path = Path(input_dir)
        landmark_files = list(input_path.glob("*_landmarks.json"))
        
        if len(landmark_files) < 2:
            print(f"Need at least 2 landmark files, found {len(landmark_files)}")
            return
            
        # Use first image as reference
        reference_file = landmark_files[0]
        
        # Compute relative poses for all other images
        for i in range(1, len(landmark_files)):
            target_file = landmark_files[i]
            self.process_image_pair(reference_file, target_file, output_dir, image_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimate camera poses from facial landmarks")
    parser.add_argument("--input", default="../data/processed", help="Input directory with landmark files")
    parser.add_argument("--output", default="../data/poses", help="Output directory for pose information")
    parser.add_argument("--height", type=int, default=512, help="Image height")
    parser.add_argument("--width", type=int, default=512, help="Image width")
    
    args = parser.parse_args()
    
    pose_estimator = CameraPoseEstimator()
    pose_estimator.process_multiview(args.input, args.output, (args.height, args.width))
