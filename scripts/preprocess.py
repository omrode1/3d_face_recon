import cv2
import numpy as np
import face_alignment
import torch
from pathlib import Path
import json
import os

class FacePreprocessor:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D, 
            device=device
        )
        self.device = device
        print(f"Using device: {device}")
        
    def detect_landmarks(self, image_path):
        """Detect 68 facial landmarks using face_alignment library"""
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Failed to load image: {image_path}")
            return None
            
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect landmarks
        try:
            landmarks = self.fa.get_landmarks(image_rgb)
            if landmarks is None or len(landmarks) == 0:
                print(f"No face detected in {image_path}")
                return None
            return landmarks[0]  # Return first face landmarks
        except Exception as e:
            print(f"Error detecting landmarks in {image_path}: {str(e)}")
            return None
    
    def align_face(self, image, landmarks):
        """Align face based on eye coordinates"""
        # Use landmarks for left and right eye centers
        left_eye = np.mean(landmarks[36:42], axis=0)
        right_eye = np.mean(landmarks[42:48], axis=0)
        
        # Calculate angle for rotation
        dY = right_eye[1] - left_eye[1]
        dX = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dY, dX))
        
        # Get center of image
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # Get rotation matrix and apply transformation
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        aligned = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)
        
        # Transform landmarks too
        ones = np.ones(shape=(len(landmarks), 1))
        landmarks_homo = np.hstack([landmarks, ones])
        transformed_landmarks = M.dot(landmarks_homo.T).T
        
        return aligned, transformed_landmarks
    
    def crop_face(self, image, landmarks, margin=0.5):
        """Crop the face with a margin around the face landmarks"""
        # Get bounding box around landmarks
        min_x, min_y = np.min(landmarks, axis=0)
        max_x, max_y = np.max(landmarks, axis=0)
        
        # Add margin
        width = max_x - min_x
        height = max_y - min_y
        min_x = max(0, min_x - margin * width)
        min_y = max(0, min_y - margin * height)
        max_x = min(image.shape[1], max_x + margin * width)
        max_y = min(image.shape[0], max_y + margin * height)
        
        # Crop image
        return image[int(min_y):int(max_y), int(min_x):int(max_x)]
    
    def process_image(self, image_path, output_dir):
        """Process a single image"""
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Failed to load image: {image_path}")
            return False
            
        # Detect landmarks
        landmarks = self.detect_landmarks(image_path)
        if landmarks is None:
            return False
            
        # Align face
        aligned_image, aligned_landmarks = self.align_face(image, landmarks)
        
        # Crop face
        cropped_image = self.crop_face(aligned_image, aligned_landmarks)
        
        # Save processed image
        output_path = Path(output_dir) / f"{Path(image_path).stem}_processed.jpg"
        cv2.imwrite(str(output_path), cropped_image)
        
        # Save landmarks
        landmark_path = Path(output_dir) / f"{Path(image_path).stem}_landmarks.json"
        with open(landmark_path, 'w') as f:
            json.dump(landmarks.tolist(), f)
            
        return True
        
    def process_directory(self, input_dir="data/raw", output_dir="data/processed"):
        """Process all images in a directory"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        image_extensions = ['.jpg', '.jpeg', '.png']
        processed_count = 0
        
        for ext in image_extensions:
            for img_path in input_path.glob(f"*{ext}"):
                print(f"Processing {img_path}...")
                if self.process_image(img_path, output_path):
                    processed_count += 1
        
        print(f"Processed {processed_count} images")
        return processed_count

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess face images")
    parser.add_argument("--input", default="../data/raw", help="Input directory")
    parser.add_argument("--output", default="../data/processed", help="Output directory")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    preprocessor = FacePreprocessor(device=args.device)
    preprocessor.process_directory(args.input, args.output)