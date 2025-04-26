import cv2
import numpy as np
import os
import time

# COCO body parts
BODY_PARTS = {
    "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
    "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
    "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
    "LEye": 15, "REar": 16, "LEar": 17, "Background": 18
}

# COCO pairs for skeleton drawing
POSE_PAIRS = [
    ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
    ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
    ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
    ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
    ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]
]

# Global variable to store the model instance for reuse
_pose_net = None

def get_pose_model():
    """
    Get or initialize the pose detection model.
    Uses a cached model if available to avoid loading on every run.
    
    Returns:
        OpenCV DNN model for pose detection
    """
    global _pose_net
    
    if _pose_net is not None:
        return _pose_net
    
    # Path to the model file
    model_path = os.path.join(os.path.dirname(__file__), "models", "graph_opt.pb")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Pose detection model not found at {model_path}")
    
    print(f"Loading pose detection model from {model_path}...")
    _pose_net = cv2.dnn.readNetFromTensorflow(model_path)
    print("Pose detection model loaded successfully")
    
    return _pose_net

def detect_pose(image, confidence_threshold=0.2):
    """
    Detect human pose in an image using TensorFlow model
    
    Args:
        image: numpy array of image (BGR format from OpenCV)
        confidence_threshold: minimum confidence for keypoint detection
        
    Returns:
        image with pose skeleton drawn
    """
    # Get a copy of the image to draw on
    result_image = image.copy()
    frameHeight, frameWidth = image.shape[:2]
    
    # Get the pose detection model
    net = get_pose_model()
    
    # Prepare input blob
    inWidth = 368
    inHeight = 368
    
    # The model expects input with mean subtraction
    inpBlob = cv2.dnn.blobFromImage(image, 1.0, (inWidth, inHeight), 
                                    (127.5, 127.5, 127.5), swapRB=True, crop=False)
    net.setInput(inpBlob)
    
    # Forward pass
    print("Running pose detection inference...")
    start = time.time()
    output = net.forward()
    end = time.time()
    inference_time = (end - start) * 1000  # Convert to milliseconds
    print(f"Pose detection inference time: {inference_time:.2f} ms")
    
    # MobileNet output [1, 57, -1, -1], we only need the first 19 elements
    output = output[:, :19, :, :]
    
    # Get output dimensions
    H = output.shape[2]
    W = output.shape[3]
    
    # Empty list to store detected keypoints
    points = []
    
    # For each keypoint
    print("Detecting keypoints...")
    for i in range(len(BODY_PARTS)):
        # Probability map
        probMap = output[0, i, :, :]
        
        # Find global maxima
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
        
        # Scale the point to fit on the original image
        x = (frameWidth * point[0]) / W
        y = (frameHeight * point[1]) / H
        
        # Add point if probability is higher than threshold
        if prob > confidence_threshold:
            cv2.circle(result_image, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            # Get keypoint name
            keypoint_name = list(BODY_PARTS.keys())[i]
            cv2.putText(result_image, keypoint_name, (int(x), int(y)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, lineType=cv2.LINE_AA)
            points.append((int(x), int(y)))
        else:
            points.append(None)
    
    # Draw skeleton
    print("Drawing skeleton...")
    for pair in POSE_PAIRS:
        partA = BODY_PARTS[pair[0]]
        partB = BODY_PARTS[pair[1]]
        
        if points[partA] and points[partB]:
            cv2.line(result_image, points[partA], points[partB], (0, 255, 0), 2)
            cv2.circle(result_image, points[partA], 5, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.circle(result_image, points[partB], 5, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
    
    # Display performance info
    cv2.putText(result_image, f'Inference time: {inference_time:.2f}ms', 
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, lineType=cv2.LINE_AA)
    
    return result_image
