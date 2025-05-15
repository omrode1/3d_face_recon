import cv2
import mediapipe as mp
import numpy as np

def extract_landmarks(image_path):
    """
    Extract 2D facial landmarks from an image using MediaPipe.
    
    Parameters:
    - image_path: Path to the input image
    
    Returns:
    - landmarks: List of (x, y) coordinates of facial landmarks
    """
    # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)
    
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return None
    
    # Convert the image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image
    results = face_mesh.process(image_rgb)
    
    # Extract landmarks
    landmarks = []
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for landmark in face_landmarks.landmark:
                x, y = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])
                landmarks.append((x, y))
    
    return landmarks

if __name__ == "__main__":
    # Example usage
    image_path = "data/processed/Front_processed.jpg"  # Update with your image path
    landmarks = extract_landmarks(image_path)
    if landmarks:
        print(f"Extracted {len(landmarks)} landmarks.")
    else:
        print("No landmarks found.") 