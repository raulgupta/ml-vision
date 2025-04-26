import torch
import numpy as np
import cv2
import os
import sys
from io import BytesIO
from PIL import Image
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
    
    with torch.inference_mode():
        # CPU path without autocast
        predictor.set_image(image_rgb)
        masks, _, _ = predictor.predict(
            point_coords=np.array([[center_x, center_y]]),
            point_labels=np.array([1])  # 1 indicates foreground
        )
    
    # Get the predicted mask
    if len(masks) == 0:
        print("No masks generated.")
        return image  # Return original image if no mask is generated
        
    mask = masks[0].astype(np.uint8) * 255
    
    # Create segmented image
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
