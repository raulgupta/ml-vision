import cv2
import numpy as np
import base64
import os
import json
from io import BytesIO
from PIL import Image
import torch
from sam2_service import segment_with_sam2
from pose_service import detect_pose

def decode_image(base64_string):
    """Convert base64 string to numpy array"""
    img_data = base64.b64decode(base64_string)
    img = Image.open(BytesIO(img_data))
    return np.array(img)

def encode_image(image_array):
    """Convert numpy array to base64 string"""
    img = Image.fromarray(image_array)
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def detect_edges(image, low_threshold=50, high_threshold=150, aperture_size=3):
    """Apply Canny edge detection to an image"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, low_threshold, high_threshold, apertureSize=aperture_size)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

def extract_features(image, method='sift', max_features=1000):
    """Extract features using SIFT or ORB"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    if method.lower() == 'sift':
        detector = cv2.SIFT_create(nfeatures=max_features)
    elif method.lower() == 'orb':
        detector = cv2.ORB_create(nfeatures=max_features)
    else:
        raise ValueError(f"Unknown method: {method}")
    keypoints, _ = detector.detectAndCompute(gray, None)
    return cv2.drawKeypoints(image.copy(), keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

def segment_image(image, confidence_threshold=0.2, yolo_results=None):
    """
    Segment image using SAM2 (Segment Anything Model 2)
    
    Parameters:
    - image: numpy array of image
    - confidence_threshold: not used with SAM2, kept for API compatibility
    - yolo_results: not used with SAM2, kept for API compatibility
    
    Returns:
    - Tuple of (segmented image with pink background, empty class counts dictionary)
    """
    try:
        print(f"Starting segment_image with SAM2, image shape: {image.shape}")
        
        # Ensure image has 3 channels
        if image.shape[2] == 4:
            print("Converting RGBA to RGB")
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        # Process with SAM2
        segmented_image = segment_with_sam2(image, use_pink_background=True)
        
        # Return empty class counts for API compatibility
        class_counts = {}
        
        print("SAM2 segmentation complete")
        return segmented_image, class_counts
        
    except Exception as e:
        print(f"Error in segment_image: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def detect_human_pose(image, confidence_threshold=0.2):
    """
    Detect human pose in an image
    
    Parameters:
    - image: numpy array of image
    - confidence_threshold: minimum confidence for keypoint detection
    
    Returns:
    - Tuple of (image with pose skeleton drawn, empty dict for API compatibility)
    """
    try:
        print(f"Starting detect_human_pose with image shape: {image.shape}")
        
        # Ensure image has 3 channels
        if image.shape[2] == 4:
            print("Converting RGBA to RGB")
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        # Process with pose detection
        result_image = detect_pose(image, confidence_threshold)
        
        # Return empty dict for API compatibility
        data = {}
        
        print("Pose detection complete")
        return result_image, data
        
    except Exception as e:
        print(f"Error in detect_human_pose: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def detect_objects(image, confidence_threshold=0.2, nms_threshold=0.4):
    """Detect objects in an image using YOLO"""
    # Load YOLO model (first time only)
    if not hasattr(detect_objects, "net"):
        weights_path = os.path.join("models", "yolov3-tiny.weights")
        config_path = os.path.join("models", "yolov3-tiny.cfg")
        classes_path = os.path.join("models", "coco.names")
        
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"YOLO weights file not found at {weights_path}. Please run download_yolo_weights.py first.")
        
        # Load network
        detect_objects.net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
        
        # Load class names
        with open(classes_path, "r") as f:
            detect_objects.classes = [line.strip() for line in f.readlines()]
        
        # Get output layer names
        detect_objects.layer_names = detect_objects.net.getLayerNames()
        detect_objects.output_layers = [detect_objects.layer_names[i - 1] for i in detect_objects.net.getUnconnectedOutLayers()]
    
    # Ensure image has 3 channels (convert RGBA to RGB if needed)
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    # Prepare image for YOLO
    height, width, _ = image.shape
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    
    # Forward pass through the network
    detect_objects.net.setInput(blob)
    outputs = detect_objects.net.forward(detect_objects.output_layers)
    
    # Process detections
    boxes = []
    confidences = []
    class_ids = []
    
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > confidence_threshold:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    # Apply non-maximum suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
    
    # Prepare results
    results = []
    result_image = image.copy()
    
    if len(indices) > 0:
        for i in indices.flatten():
            box = boxes[i]
            x, y, w, h = box
            label = detect_objects.classes[class_ids[i]]
            confidence = confidences[i]
            
            # Add to results
            results.append({
                "label": label,
                "confidence": float(confidence),  # Convert numpy float to Python float for JSON serialization
                "bbox": [int(x), int(y), int(w), int(h)]  # Convert numpy int to Python int
            })
            
            # Draw bounding box
            color = (0, 255, 0)  # Green
            cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 2)
            
            # Draw label
            text = f"{label}: {confidence:.2f}"
            cv2.putText(result_image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return result_image, results
