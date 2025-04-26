'use client';

import { useState, useEffect, useRef } from 'react';
import Link from 'next/link';
import PersistentInput from '../components/PersistentInput';
import ImageGenTitle from '../components/ImageGenTitle';
import CodeTheaterModal from '../components/CodeTheaterModal';
import { cvService } from '../components/CVService';

// Style constants (matching AgentInterface)
const MAX_WIDTH = 'max-w-[580px]';
const BASE_CONTAINER = `w-full ${MAX_WIDTH}`;
const GLASS_EFFECT = 'bg-white/[0.04] backdrop-blur-[8px]';
const HOVER_BORDER = 'border border-white/5 hover:border-white/10';

// Code snippets for the theater modal
const edgeDetectionCode = `# Canny Edge Detection with OpenCV
import cv2
import numpy as np

def detect_edges(image, low_threshold=50, high_threshold=150, aperture_size=3):
    """
    Apply Canny edge detection to an image
    
    Parameters:
    - image: numpy array of image
    - low_threshold: lower threshold for hysteresis (default: 50)
    - high_threshold: upper threshold for hysteresis (default: 150)
    - aperture_size: aperture size for Sobel operator (default: 3)
    
    Returns:
    - edge image as numpy array
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Canny edge detector
    edges = cv2.Canny(blurred, 
                     low_threshold, 
                     high_threshold, 
                     apertureSize=aperture_size)
    
    # Convert to 3-channel for display
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    
    return edges_rgb`;

const featureExtractionCode = `# Feature Extraction with OpenCV
import cv2
import numpy as np

def extract_features(image, method='sift', max_features=1000):
    """
    Extract features from an image using SIFT or ORB
    
    Parameters:
    - image: numpy array of image
    - method: 'sift' or 'orb' (default: 'sift')
    - max_features: maximum number of features to detect (default: 1000)
    
    Returns:
    - Visualization image with keypoints drawn
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    if method.lower() == 'sift':
        # SIFT detector (scale-invariant)
        detector = cv2.SIFT_create(nfeatures=max_features)
    elif method.lower() == 'orb':
        # ORB detector (faster, rotation-invariant)
        detector = cv2.ORB_create(nfeatures=max_features)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Detect keypoints and compute descriptors
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    
    # Create visualization
    vis_image = cv2.drawKeypoints(
        image.copy(), 
        keypoints, 
        None, 
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    
    return vis_image`;

const poseDetectionCode = `# Human Pose Detection with TensorFlow and OpenCV
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
    
    # Load the TensorFlow model
    model_path = os.path.join("models", "graph_opt.pb")
    net = cv2.dnn.readNetFromTensorflow(model_path)
    
    # Prepare input blob
    inWidth = 368
    inHeight = 368
    
    # The model expects input with mean subtraction
    inpBlob = cv2.dnn.blobFromImage(image, 1.0, (inWidth, inHeight), 
                                    (127.5, 127.5, 127.5), swapRB=True, crop=False)
    net.setInput(inpBlob)
    
    # Forward pass
    start = time.time()
    output = net.forward()
    inference_time = (time.time() - start) * 1000  # Convert to milliseconds
    
    # MobileNet output [1, 57, -1, -1], we only need the first 19 elements
    output = output[:, :19, :, :]
    
    # Get output dimensions
    H = output.shape[2]
    W = output.shape[3]
    
    # Empty list to store detected keypoints
    points = []
    
    # For each keypoint
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
    
    return result_image`;

const segmentationCode = `# Image Segmentation with SAM2 (Segment Anything Model 2) and PyTorch
import torch
import numpy as np
import cv2
import os
from sam2.sam2_image_predictor import SAM2ImagePredictor

def segment_with_sam2(image, use_pink_background=True):
    """
    Segment the image using SAM2 small model from Hugging Face.
    
    Args:
        image: numpy array of image (BGR format from OpenCV)
        use_pink_background: If True, use pink color for background instead of black
        
    Returns:
        segmented_image: numpy array of segmented image with pink background
    """
    # Convert BGR to RGB for SAM2
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Force CPU mode since CUDA is not available
    print("Using CPU for inference (CUDA not available)")
    torch.cuda.is_available = lambda: False  # Override CUDA availability check
    
    # Set environment variable to disable CUDA
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    # Load SAM2 small model from Hugging Face
    print("Loading SAM2 small model from Hugging Face...")
    
    # Create cache directory if it doesn't exist
    cache_dir = os.path.join(os.path.dirname(__file__), "models", "sam2_cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Explicitly pass CPU device to from_pretrained
    predictor = SAM2ImagePredictor.from_pretrained(
        "facebook/sam2-hiera-base-plus",
        device="cpu",
        cache_dir=cache_dir
    )
    
    # Process image
    print("Generating segmentation mask...")
    
    # Use center point as prompt
    h, w = image.shape[:2]
    center_x, center_y = w // 2, h // 2
    print(f"Using center point prompt at ({center_x}, {center_y})")
    
    # PyTorch inference mode for efficiency
    with torch.inference_mode():
        # Set the image in the predictor
        predictor.set_image(image_rgb)
        
        # Predict mask using center point prompt
        # point_coords: array of point coordinates
        # point_labels: array of point labels (1 for foreground, 0 for background)
        masks, _, _ = predictor.predict(
            point_coords=np.array([[center_x, center_y]]),
            point_labels=np.array([1])  # 1 indicates foreground
        )
    
    # Get the predicted mask
    if len(masks) == 0:
        print("No masks generated.")
        return image  # Return original image if no mask is generated
        
    mask = masks[0].astype(np.uint8) * 255
    
    # Create segmented image with pink background
    if use_pink_background:
        # Create a result image filled with pink (BGR format)
        result = np.full_like(image, [203, 192, 255], dtype=np.uint8)
        
        # Copy the foreground pixels directly from the original image
        # Only where the mask is non-zero
        np.copyto(result, image, where=(mask[:, :, np.newaxis] > 0))
        
        segmented = result
    else:
        # Just apply mask to original image (black background)
        segmented = cv2.bitwise_and(image, image, mask=mask)
    
    return segmented

def segment_image(image):
    """
    Segment image using SAM2 (Segment Anything Model 2)
    
    Parameters:
    - image: numpy array of image
    
    Returns:
    - Segmented image with pink background
    """
    try:
        print(f"Starting segment_image with SAM2, image shape: {image.shape}")
        
        # Ensure image has 3 channels
        if image.shape[2] == 4:
            print("Converting RGBA to RGB")
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        # Process with SAM2
        segmented_image = segment_with_sam2(image, use_pink_background=True)
        
        print("SAM2 segmentation complete")
        return segmented_image
        
    except Exception as e:
        print(f"Error in segment_image: {str(e)}")
        import traceback
        traceback.print_exc()
        raise`;

const objectDetectionCode = `# Object Detection with YOLO and OpenCV
import cv2
import numpy as np
import os

def detect_objects(image, confidence_threshold=0.2, nms_threshold=0.4):
    """
    Detect objects in an image using YOLO
    
    Parameters:
    - image: numpy array of image
    - confidence_threshold: minimum confidence for detection (default: 0.2)
    - nms_threshold: non-maximum suppression threshold (default: 0.4)
    
    Returns:
    - Tuple of (image with bounding boxes drawn, list of detected objects)
    """
    # Load YOLO model (first time only)
    if not hasattr(detect_objects, "net"):
        # Check if weights file exists
        weights_path = os.path.join("models", "yolov3-tiny.weights")
        config_path = os.path.join("models", "yolov3-tiny.cfg")
        classes_path = os.path.join("models", "coco.names")
        
        # Load network
        detect_objects.net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
        
        # Load class names
        with open(classes_path, "r") as f:
            detect_objects.classes = [line.strip() for line in f.readlines()]
        
        # Get output layer names
        detect_objects.layer_names = detect_objects.net.getLayerNames()
        detect_objects.output_layers = [detect_objects.layer_names[i - 1] for i in detect_objects.net.getUnconnectedOutLayers()]
    
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
                "confidence": float(confidence),
                "bbox": [int(x), int(y), int(w), int(h)]
            })
            
            # Draw bounding box
            color = (0, 255, 0)  # Green
            cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 2)
            
            # Draw label
            text = f"{label}: {confidence:.2f}"
            cv2.putText(result_image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return result_image, results`;

export default function Home() {
  const [prompt, setPrompt] = useState('');
  const [negativePrompt, setNegativePrompt] = useState('');
  const [imageUrl, setImageUrl] = useState('');
  const [revisedPrompt, setRevisedPrompt] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState('');
  const [activeImage, setActiveImage] = useState('original');
  const [processedImages, setProcessedImages] = useState<{[key: string]: string}>({});
  const [codeModalOpen, setCodeModalOpen] = useState(false);
  const [currentAlgorithm, setCurrentAlgorithm] = useState<'edges' | 'features' | 'objects' | 'segmentation' | 'pose'>('edges');
  const [imageDimensions, setImageDimensions] = useState<{width: number, height: number} | null>(null);
  const originalImageRef = useRef<HTMLImageElement>(null);
  
  // Edge detection parameters
  const [edgeParams, setEdgeParams] = useState({
    lowThreshold: 50,
    highThreshold: 150,
    apertureSize: 3
  });
  
  // Feature extraction parameters
  const [featureParams, setFeatureParams] = useState({
    method: 'sift' as 'sift' | 'orb',
    maxFeatures: 1000
  });
  
  // Object detection parameters - 0.2 threshold was decided as the standard value
  // for better detection of smaller/less confident objects
  const [objectParams, setObjectParams] = useState({
    confidenceThreshold: 0.2,
    nmsThreshold: 0.4
  });
  
  // Pose detection parameters
  const [poseParams, setPoseParams] = useState({
    confidenceThreshold: 0.2
  });
  
  // Generate image using OpenAI API
  const generateImage = async (userPrompt: string) => {
    setIsProcessing(true);
    setError('');
    setPrompt(userPrompt);
    
    try {
      const response = await fetch('/api/generate-image', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          prompt: userPrompt, 
          negativePrompt 
        })
      });
      
      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.error || 'Failed to generate image');
      }
      
      setImageUrl(data.imageUrl);
      setRevisedPrompt(data.revisedPrompt || '');
      setActiveImage('original');
      setProcessedImages({});
      setImageDimensions(null);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setIsProcessing(false);
    }
  };
  
  // State for segmentation data
  const [segmentationData, setSegmentationData] = useState<{[className: string]: {pixel_count: number, percentage: number}} | null>(null);
  
  // Apply CV processing to the image
  const applyProcessing = async (
    type: 'edges' | 'features' | 'objects' | 'segmentation' | 'pose', 
    forceReprocess = false,
    dataset?: 'voc' | 'coco'
  ) => {
    if (!imageUrl) return;
    
    // If we already processed this type and not forcing reprocess, just switch to it
    if (!forceReprocess && processedImages[type]) {
      setActiveImage(type);
      return;
    }
    
    setIsProcessing(true);
    
    try {
      // Create a new image element to ensure it's fully loaded
      const img = new Image();
      img.crossOrigin = 'anonymous';
      
      // Wait for the image to load
      await new Promise((resolve, reject) => {
        img.onload = () => {
          // Store the image dimensions
          setImageDimensions({
            width: img.naturalWidth,
            height: img.naturalHeight
          });
          resolve(null);
        };
        img.onerror = (e) => {
          console.error('Image load error:', e);
          reject(new Error('Failed to load image due to CORS restrictions. This is a limitation of the demo.'));
        };
        img.src = imageUrl;
      });
      
      // Ensure we have valid dimensions
      if (!img.naturalWidth || !img.naturalHeight) {
        throw new Error('Could not determine image dimensions');
      }
      
      const canvas = document.createElement('canvas');
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      const ctx = canvas.getContext('2d')!;
      
      // Draw the original image
      ctx.drawImage(img, 0, 0);
      
      // Apply the selected processing
      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      let processedData;
      
      if (type === 'edges') {
        // Use our enhanced CV service for edge detection
        const result = await cvService.detectEdges(imageData, edgeParams);
        if (result.error) {
          throw new Error(result.error);
        }
        processedData = result.outputImage;
      } else if (type === 'features') {
        // Use our enhanced CV service for feature extraction
        const result = await cvService.extractFeatures(imageData, featureParams);
        if (result.error) {
          throw new Error(result.error);
        }
        processedData = result.outputImage;
      } else if (type === 'objects') {
        // Use our enhanced CV service for object detection
        const result = await cvService.detectObjects(imageData, objectParams);
        if (result.error) {
          throw new Error(result.error);
        }
        processedData = result.outputImage;
      } else if (type === 'segmentation') {
        // Use our enhanced CV service for image segmentation
        const result = await cvService.segmentImage(imageData, {
          confidenceThreshold: 0.2,
          useYolo: true,
          yoloConfidence: 0.2
        });
        if (result.error) {
          throw new Error(result.error);
        }
        processedData = result.outputImage;
        // Store segmentation data for display
        setSegmentationData(result.segmentationData || null);
      } else if (type === 'pose') {
        // Use our enhanced CV service for pose detection
        const result = await cvService.detectPose(imageData, poseParams);
        if (result.error) {
          throw new Error(result.error);
        }
        processedData = result.outputImage;
      } else {
        throw new Error('Unknown processing type');
      }
      
      if (!processedData) {
        throw new Error('Processing failed to return image data');
      }
      
      ctx.putImageData(processedData, 0, 0);
      
      // Save the processed image
      const processedUrl = canvas.toDataURL('image/png');
      setProcessedImages(prev => ({
        ...prev,
        [type]: processedUrl
      }));
      setActiveImage(type);
    } catch (err: any) {
      console.error('Error processing image:', err);
      setError('Failed to process image. Please try again.');
    } finally {
      setIsProcessing(false);
    }
  };
  
  // Edge detection algorithm (Sobel)
  const detectEdges = (imageData: ImageData): ImageData => {
    const { width, height, data } = imageData;
    const output = new ImageData(width, height);
    const outputData = output.data;
    
    for (let y = 1; y < height - 1; y++) {
      for (let x = 1; x < width - 1; x++) {
        const idx = (y * width + x) * 4;
        
        // For each color channel
        for (let c = 0; c < 3; c++) {
          // Sobel kernels
          const gx = 
            -1 * data[((y-1) * width + (x-1)) * 4 + c] +
            -2 * data[((y) * width + (x-1)) * 4 + c] +
            -1 * data[((y+1) * width + (x-1)) * 4 + c] +
            1 * data[((y-1) * width + (x+1)) * 4 + c] +
            2 * data[((y) * width + (x+1)) * 4 + c] +
            1 * data[((y+1) * width + (x+1)) * 4 + c];
          
          const gy = 
            -1 * data[((y-1) * width + (x-1)) * 4 + c] +
            -2 * data[((y-1) * width + (x)) * 4 + c] +
            -1 * data[((y-1) * width + (x+1)) * 4 + c] +
            1 * data[((y+1) * width + (x-1)) * 4 + c] +
            2 * data[((y+1) * width + (x)) * 4 + c] +
            1 * data[((y+1) * width + (x+1)) * 4 + c];
          
          const magnitude = Math.sqrt(gx * gx + gy * gy);
          
          // Threshold
          outputData[idx + c] = magnitude > 50 ? 255 : 0;
        }
        
        // Alpha channel
        outputData[idx + 3] = 255;
      }
    }
    
    return output;
  };
  
  // Feature extraction visualization
  const extractFeatures = (imageData: ImageData): ImageData => {
    const { width, height, data } = imageData;
    const output = new ImageData(width, height);
    const outputData = output.data;
    
    // First copy the original image (with reduced opacity)
    for (let i = 0; i < data.length; i += 4) {
      outputData[i] = data[i] * 0.7;     // R
      outputData[i + 1] = data[i + 1] * 0.7; // G
      outputData[i + 2] = data[i + 2] * 0.7; // B
      outputData[i + 3] = data[i + 3];   // A
    }
    
    // Find "interesting" points (high contrast areas)
    const points = [];
    const blockSize = 16;
    
    for (let y = blockSize; y < height - blockSize; y += blockSize) {
      for (let x = blockSize; x < width - blockSize; x += blockSize) {
        // Calculate local variance as a simple feature detector
        let sum = 0;
        let sumSq = 0;
        let count = 0;
        
        for (let dy = -blockSize/2; dy < blockSize/2; dy++) {
          for (let dx = -blockSize/2; dx < blockSize/2; dx++) {
            const idx = ((y + dy) * width + (x + dx)) * 4;
            const gray = (data[idx] + data[idx + 1] + data[idx + 2]) / 3;
            sum += gray;
            sumSq += gray * gray;
            count++;
          }
        }
        
        const mean = sum / count;
        const variance = (sumSq / count) - (mean * mean);
        
        if (variance > 500) { // Threshold for "interesting" points
          points.push({ x, y });
        }
      }
    }
    
    // Draw feature points
    for (const point of points) {
      const { x, y } = point;
      
      // Draw a small circle
      for (let dy = -3; dy <= 3; dy++) {
        for (let dx = -3; dx <= 3; dx++) {
          if (dx*dx + dy*dy <= 9) {
            const px = x + dx;
            const py = y + dy;
            
            if (px >= 0 && px < width && py >= 0 && py < height) {
              const idx = (py * width + px) * 4;
              outputData[idx] = 255;     // R
              outputData[idx + 1] = 50;  // G
              outputData[idx + 2] = 50;  // B
              outputData[idx + 3] = 255; // A
            }
          }
        }
      }
    }
    
    return output;
  };
  
  // Show code theater and set current algorithm
  const showCodeTheater = (algorithm: 'edges' | 'features' | 'objects' | 'segmentation' | 'pose') => {
    setCurrentAlgorithm(algorithm);
    setCodeModalOpen(true);
  };
  
  // Execute the selected algorithm
  const executeAlgorithm = () => {
    // Always force reprocessing when executing from the Code Theater
    applyProcessing(currentAlgorithm, true);
  };
  
  return (
    <main className="min-h-screen w-full relative overflow-hidden">
      {/* Military Background Pattern */}
      <div className="absolute inset-0 military-gradient">
        <div className="absolute inset-0 military-mesh opacity-20" />
        <div className="absolute inset-0 military-mesh opacity-10 scale-150 rotate-45" />
        <div className="absolute inset-0 military-mesh opacity-5 scale-200 -rotate-45" />
      </div>

      <div className="container mx-auto px-4 py-8">
        <div className="mt-40 md:mt-48">
          <div className="max-w-3xl mx-auto">
            {/* Title and CV LAB button */}
            <div className="relative mb-12">
              {/* CV LAB button - absolute positioned in top right */}
              <div className="absolute top-2 right-0 z-10">
                <Link 
                  href="/cv"
                  className="px-4 py-1.5 md:px-6 md:py-2 bg-white/[0.02] backdrop-blur-sm border border-white/[0.03] rounded-lg text-base md:text-lg text-white/40 hover:text-white/90 hover:bg-white/[0.04] transition-all duration-300"
                >
                  CV LAB
                </Link>
              </div>
              
              {/* Title - centered */}
              <div className="w-full flex justify-center">
                <ImageGenTitle />
              </div>
            </div>
            
            {/* Main Content */}
            <div className="space-y-8">
              {/* Prompt Input */}
              <div className="w-full">
                <PersistentInput
                  onSubmit={generateImage}
                  isProcessing={isProcessing}
                  initialValue={prompt}
                  placeholders={[
                    "A futuristic city with flying cars...",
                    "A photorealistic portrait of a cyberpunk character...",
                    "An astronaut riding a horse on Mars...",
                    "A steampunk-inspired coffee machine...",
                    "A serene Japanese garden in autumn...",
                    "A hyperrealistic close-up of a butterfly wing...",
                    "An underwater city with bioluminescent buildings...",
                    "A fantasy landscape with floating islands..."
                  ]}
                />
              </div>
              
              {/* Negative Prompt Input */}
              <div className={`${BASE_CONTAINER} mx-auto`}>
                <div className={`${GLASS_EFFECT} rounded-xl p-4`}>
                  <label className="block text-green-500 text-sm font-mono mb-2">
                    NEGATIVE PROMPT:
                  </label>
                  <input
                    type="text"
                    value={negativePrompt}
                    onChange={(e) => setNegativePrompt(e.target.value)}
                    placeholder="Elements to avoid in the image..."
                    className="w-full px-3 py-2 bg-white/[0.02] text-white/80 rounded-lg border text-sm border-white/[0.05] focus:outline-none focus:border-white/10"
                    disabled={isProcessing}
                  />
                </div>
              </div>
              
              {/* Error Display */}
              {error && (
                <div className={`${BASE_CONTAINER} mx-auto p-3 bg-red-500/10 border border-red-500/20 rounded-lg`}>
                  <p className="text-red-400 text-sm">{error}</p>
                </div>
              )}
              
              {/* Generated Image Display */}
              {imageUrl && (
                <div className={`${BASE_CONTAINER} mx-auto ${GLASS_EFFECT} rounded-xl p-4 ${HOVER_BORDER}`}>
                  <div className="mb-4">
                    <h3 className="text-white/70 font-medium mb-2">Generated Image:</h3>
                    {revisedPrompt && (
                      <p className="text-white/50 text-sm italic mb-4">
                        Refined prompt: {revisedPrompt}
                      </p>
                    )}
                    
                    {/* Image Dimensions (optional debug info) */}
                    {imageDimensions && (
                      <p className="text-white/30 text-xs mb-2">
                        Dimensions: {imageDimensions.width}×{imageDimensions.height}px
                      </p>
                    )}
                    
                    {/* Image */}
                    <div className="relative aspect-square w-full max-w-[1024px] mx-auto overflow-hidden rounded-lg border border-white/10">
                      {activeImage === 'original' ? (
                        <img 
                          ref={originalImageRef}
                          src={imageUrl}
                          alt={prompt}
                          className="w-full h-full object-cover"
                          onLoad={(e) => {
                            const img = e.currentTarget;
                            setImageDimensions({
                              width: img.naturalWidth,
                              height: img.naturalHeight
                            });
                          }}
                        />
                      ) : (
                        <img 
                          src={processedImages[activeImage]}
                          alt={`${prompt} (${activeImage})`}
                          className="w-full h-full object-cover"
                        />
                      )}
                      
                      {/* Loading Overlay */}
                      {isProcessing && (
                        <div className="absolute inset-0 bg-black/50 flex items-center justify-center">
                          <div className="text-white">Processing...</div>
                        </div>
                      )}
                    </div>
                  </div>
                  
                  {/* CV Processing Options */}
                  <div className="flex flex-col gap-2">
                    {/* Top row - Original, Edge Detection, Feature Extraction */}
                    <div className="flex flex-wrap gap-2 justify-start">
                      <button 
                        onClick={() => setActiveImage('original')}
                        className="px-4 py-2 rounded-lg bg-white/[0.02] hover:bg-white/[0.04] text-white/70 hover:text-white/90 transition-all duration-300 flex items-center gap-2"
                      >
                        <div className={`w-2 h-2 rounded-full ${
                          activeImage === 'original' 
                            ? 'bg-[#ff66a8]' 
                            : 'bg-white/20'
                        }`} />
                        Original
                      </button>
                      <button 
                        onClick={() => {
                          // If we're already showing edges, apply processing with force=true
                          // Otherwise show the code theater
                          if (activeImage === 'edges') {
                            applyProcessing('edges', true);
                          } else {
                            showCodeTheater('edges');
                          }
                        }}
                        className="px-2 py-2 rounded-lg bg-white/[0.02] hover:bg-white/[0.04] text-white/70 hover:text-white/90 transition-all duration-300 flex items-center gap-2"
                        disabled={isProcessing}
                      >
                        <div className={`w-2 h-2 rounded-full ${
                          activeImage === 'edges' 
                            ? 'bg-[#ff66a8]' 
                            : 'bg-white/20'
                        }`} />
                        Edge Detection
                      </button>
                      <button 
                        onClick={() => {
                          // If we're already showing features, apply processing with force=true
                          // Otherwise show the code theater
                          if (activeImage === 'features') {
                            applyProcessing('features', true);
                          } else {
                            showCodeTheater('features');
                          }
                        }}
                        className="px-4 py-2 rounded-lg bg-white/[0.02] hover:bg-white/[0.04] text-white/70 hover:text-white/90 transition-all duration-300 flex items-center gap-2"
                        disabled={isProcessing}
                      >
                        <div className={`w-2 h-2 rounded-full ${
                          activeImage === 'features' 
                            ? 'bg-[#ff66a8]' 
                            : 'bg-white/20'
                        }`} />
                        Feature Extraction
                      </button>
                    </div>
                    
                    {/* Bottom row - Object Detection and Image Segmentation */}
                    <div className="flex flex-wrap gap-2 justify-start">
                      <button 
                        onClick={() => {
                          // If we're already showing objects, apply processing with force=true
                          // Otherwise show the code theater
                          if (activeImage === 'objects') {
                            applyProcessing('objects', true);
                          } else {
                            showCodeTheater('objects');
                          }
                        }}
                        className="px-4 py-2 rounded-lg bg-white/[0.02] hover:bg-white/[0.04] text-white/70 hover:text-white/90 transition-all duration-300 flex items-center gap-2"
                        disabled={isProcessing}
                      >
                        <div className={`w-2 h-2 rounded-full ${
                          activeImage === 'objects' 
                            ? 'bg-[#ff66a8]' 
                            : 'bg-white/20'
                        }`} />
                        Object Detection
                      </button>
                      <button 
                        onClick={() => {
                          // If we're already showing segmentation, apply processing with force=true
                          // Otherwise show the code theater
                          if (activeImage === 'segmentation') {
                            applyProcessing('segmentation', true);
                          } else {
                            showCodeTheater('segmentation');
                          }
                        }}
                        className="px-4 py-2 rounded-lg bg-white/[0.02] hover:bg-white/[0.04] text-white/70 hover:text-white/90 transition-all duration-300 flex items-center gap-2"
                        disabled={isProcessing}
                      >
                        <div className={`w-2 h-2 rounded-full ${
                          activeImage === 'segmentation' 
                            ? 'bg-[#ff66a8]' 
                            : 'bg-white/20'
                        }`} />
                        Image Segmentation
                      </button>
                      <button 
                        onClick={() => {
                          // If we're already showing pose detection, apply processing with force=true
                          // Otherwise show the code theater
                          if (activeImage === 'pose') {
                            applyProcessing('pose', true);
                          } else {
                            showCodeTheater('pose');
                          }
                        }}
                        className="px-4 py-2 rounded-lg bg-white/[0.02] hover:bg-white/[0.04] text-white/70 hover:text-white/90 transition-all duration-300 flex items-center gap-2"
                        disabled={isProcessing}
                      >
                        <div className={`w-2 h-2 rounded-full ${
                          activeImage === 'pose' 
                            ? 'bg-[#ff66a8]' 
                            : 'bg-white/20'
                        }`} />
                        Pose Detection
                      </button>
                    </div>
                  </div>
                  
                  {/* Edge Detection Parameters */}
                  {activeImage === 'edges' && (
                    <div className="mt-4 p-4 bg-white/[0.02] rounded-lg">
                      <h4 className="text-white/70 mb-2">Edge Detection Parameters</h4>
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        <div>
                          <label className="block text-white/50 text-sm mb-1">
                            Low Threshold: {edgeParams.lowThreshold}
                          </label>
                          <input
                            type="range"
                            min="0"
                            max="255"
                            value={edgeParams.lowThreshold}
                            onChange={(e) => setEdgeParams({
                              ...edgeParams,
                              lowThreshold: parseInt(e.target.value)
                            })}
                            className="w-full"
                          />
                        </div>
                        <div>
                          <label className="block text-white/50 text-sm mb-1">
                            High Threshold: {edgeParams.highThreshold}
                          </label>
                          <input
                            type="range"
                            min="0"
                            max="255"
                            value={edgeParams.highThreshold}
                            onChange={(e) => setEdgeParams({
                              ...edgeParams,
                              highThreshold: parseInt(e.target.value)
                            })}
                            className="w-full"
                          />
                        </div>
                        <div>
                          <label className="block text-white/50 text-sm mb-1">
                            Aperture Size
                          </label>
                          <select
                            value={edgeParams.apertureSize}
                            onChange={(e) => setEdgeParams({
                              ...edgeParams,
                              apertureSize: parseInt(e.target.value)
                            })}
                            className="w-full bg-white/[0.05] border border-white/10 rounded px-2 py-1 text-white/70"
                          >
                            <option value="3">3×3</option>
                            <option value="5">5×5</option>
                            <option value="7">7×7</option>
                          </select>
                        </div>
                      </div>
                      <button
                        onClick={() => {
                          // Clear the cached processed image to force reprocessing
                          setProcessedImages(prev => {
                            const newImages = {...prev};
                            delete newImages['edges'];
                            return newImages;
                          });
                          // Force reprocessing regardless of cache state
                          applyProcessing('edges', true);
                        }}
                        className="mt-3 px-4 py-2 bg-white/[0.05] hover:bg-white/[0.1] rounded text-white/70"
                        disabled={isProcessing}
                      >
                        Reprocess with New Parameters
                      </button>
                    </div>
                  )}
                  
                  {/* Feature Extraction Parameters */}
                  {activeImage === 'features' && (
                    <div className="mt-4 p-4 bg-white/[0.02] rounded-lg">
                      <h4 className="text-white/70 mb-2">Feature Extraction Parameters</h4>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div>
                          <label className="block text-white/50 text-sm mb-1">
                            Method
                          </label>
                          <select
                            value={featureParams.method}
                            onChange={(e) => setFeatureParams({
                              ...featureParams,
                              method: e.target.value as 'sift' | 'orb'
                            })}
                            className="w-full bg-white/[0.05] border border-white/10 rounded px-2 py-1 text-white/70"
                          >
                            <option value="sift">SIFT (Scale-Invariant Feature Transform)</option>
                            <option value="orb">ORB (Oriented FAST and Rotated BRIEF)</option>
                          </select>
                        </div>
                        <div>
                          <label className="block text-white/50 text-sm mb-1">
                            Max Features: {featureParams.maxFeatures}
                          </label>
                          <input
                            type="range"
                            min="100"
                            max="2000"
                            step="100"
                            value={featureParams.maxFeatures}
                            onChange={(e) => setFeatureParams({
                              ...featureParams,
                              maxFeatures: parseInt(e.target.value)
                            })}
                            className="w-full"
                          />
                        </div>
                      </div>
                      <button
                        onClick={() => {
                          // Clear the cached processed image to force reprocessing
                          setProcessedImages(prev => {
                            const newImages = {...prev};
                            delete newImages['features'];
                            return newImages;
                          });
                          // Force reprocessing
                          applyProcessing('features', true);
                        }}
                        className="mt-3 px-4 py-2 bg-white/[0.05] hover:bg-white/[0.1] rounded text-white/70"
                        disabled={isProcessing}
                      >
                        Reprocess with New Parameters
                      </button>
                    </div>
                  )}
                  
                  {/* Object Detection Parameters */}
                  {activeImage === 'objects' && (
                    <div className="mt-4 p-4 bg-white/[0.02] rounded-lg">
                      <h4 className="text-white/70 mb-2">Object Detection Parameters</h4>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div>
                          <label className="block text-white/50 text-sm mb-1">
                            Confidence Threshold: {objectParams.confidenceThreshold.toFixed(2)}
                          </label>
                          <input
                            type="range"
                            min="0.1"
                            max="0.9"
                            step="0.05"
                            value={objectParams.confidenceThreshold}
                            onChange={(e) => setObjectParams({
                              ...objectParams,
                              confidenceThreshold: parseFloat(e.target.value)
                            })}
                            className="w-full"
                          />
                          <p className="text-white/40 text-xs mt-1">
                            Higher values show fewer, more confident detections
                          </p>
                        </div>
                        <div>
                          <label className="block text-white/50 text-sm mb-1">
                            NMS Threshold: {objectParams.nmsThreshold.toFixed(2)}
                          </label>
                          <input
                            type="range"
                            min="0.1"
                            max="0.9"
                            step="0.05"
                            value={objectParams.nmsThreshold}
                            onChange={(e) => setObjectParams({
                              ...objectParams,
                              nmsThreshold: parseFloat(e.target.value)
                            })}
                            className="w-full"
                          />
                          <p className="text-white/40 text-xs mt-1">
                            Controls overlap between bounding boxes
                          </p>
                        </div>
                      </div>
                      <button
                        onClick={() => {
                          // Clear the cached processed image to force reprocessing
                          setProcessedImages(prev => {
                            const newImages = {...prev};
                            delete newImages['objects'];
                            return newImages;
                          });
                          // Force reprocessing
                          applyProcessing('objects', true);
                        }}
                        className="mt-3 px-4 py-2 bg-white/[0.05] hover:bg-white/[0.1] rounded text-white/70"
                        disabled={isProcessing}
                      >
                        Reprocess with New Parameters
                      </button>
                    </div>
                  )}
                  
                  {/* Image Segmentation Parameters */}
                  {activeImage === 'segmentation' && (
                    <div className="mt-4 p-4 bg-white/[0.02] rounded-lg">
                      <h4 className="text-white/70 mb-2">Image Segmentation Results</h4>
                      <p className="text-white/60 text-sm mb-4">
                        SAM2 (Segment Anything Model 2) is used to segment the foreground object from the background.
                        The background is colored pink for better visualization.
                      </p>
                      
                      <div className="mb-4">
                        <p className="text-white/50 text-sm">
                          The model uses PyTorch and automatically identifies the main object in the image using a center point prompt.
                        </p>
                      </div>
                      
                      <button
                        onClick={() => {
                          // Clear the cached processed image to force reprocessing
                          setProcessedImages(prev => {
                            const newImages = {...prev};
                            delete newImages['segmentation'];
                            return newImages;
                          });
                          // Force reprocessing
                          applyProcessing('segmentation', true);
                        }}
                        className="mt-3 px-4 py-2 bg-white/[0.05] hover:bg-white/[0.1] rounded text-white/70"
                        disabled={isProcessing}
                      >
                        Reprocess Image
                      </button>
                    </div>
                  )}
                  
                  {/* Pose Detection Parameters */}
                  {activeImage === 'pose' && (
                    <div className="mt-4 p-4 bg-white/[0.02] rounded-lg">
                      <h4 className="text-white/70 mb-2">Pose Detection Results</h4>
                      <p className="text-white/60 text-sm mb-4">
                        Human pose detection using TensorFlow model to identify body keypoints and draw a skeleton.
                      </p>
                      
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div>
                          <label className="block text-white/50 text-sm mb-1">
                            Confidence Threshold: {poseParams.confidenceThreshold.toFixed(2)}
                          </label>
                          <input
                            type="range"
                            min="0.1"
                            max="0.9"
                            step="0.05"
                            value={poseParams.confidenceThreshold}
                            onChange={(e) => setPoseParams({
                              ...poseParams,
                              confidenceThreshold: parseFloat(e.target.value)
                            })}
                            className="w-full"
                          />
                          <p className="text-white/40 text-xs mt-1">
                            Higher values show fewer, more confident keypoints
                          </p>
                        </div>
                      </div>
                      
                      <button
                        onClick={() => {
                          // Clear the cached processed image to force reprocessing
                          setProcessedImages(prev => {
                            const newImages = {...prev};
                            delete newImages['pose'];
                            return newImages;
                          });
                          // Force reprocessing
                          applyProcessing('pose', true);
                        }}
                        className="mt-3 px-4 py-2 bg-white/[0.05] hover:bg-white/[0.1] rounded text-white/70"
                        disabled={isProcessing}
                      >
                        Reprocess Image
                      </button>
                    </div>
                  )}
                </div>
              )}
              
              {/* Empty State */}
              {!imageUrl && !error && !isProcessing && (
                <div className={`${BASE_CONTAINER} mx-auto ${GLASS_EFFECT} rounded-xl p-8 text-center`}>
                  <p className="text-white/40 mb-2">No image generated yet</p>
                  <p className="text-white/30 text-sm">Enter a prompt to generate an image</p>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
      
      {/* Code Theater Modal */}
      <CodeTheaterModal
        isOpen={codeModalOpen}
        onClose={() => setCodeModalOpen(false)}
        title={
          currentAlgorithm === 'edges' 
            ? 'Edge Detection Algorithm' 
            : currentAlgorithm === 'features'
              ? 'Feature Extraction Algorithm'
              : currentAlgorithm === 'objects'
                ? 'Object Detection Algorithm'
                : currentAlgorithm === 'pose'
                  ? 'Pose Detection Algorithm'
                  : 'Image Segmentation Algorithm'
        }
        codeSnippet={
          currentAlgorithm === 'edges' 
            ? edgeDetectionCode 
            : currentAlgorithm === 'features'
              ? featureExtractionCode
              : currentAlgorithm === 'objects'
                ? objectDetectionCode
                : currentAlgorithm === 'pose'
                  ? poseDetectionCode
                  : segmentationCode
        }
        onExecute={executeAlgorithm}
      />
    </main>
  );
}
