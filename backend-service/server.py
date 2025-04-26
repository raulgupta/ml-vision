from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from cv_service import decode_image, encode_image, detect_edges, extract_features, detect_objects, segment_image, detect_human_pose
from benchmark_service import run_benchmarks
import time

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class EdgeDetectionRequest(BaseModel):
    image: str  # base64 encoded image
    low_threshold: int = 50
    high_threshold: int = 150
    aperture_size: int = 3

class FeatureExtractionRequest(BaseModel):
    image: str  # base64 encoded image
    method: str = 'sift'  # 'sift' or 'orb'
    max_features: int = 1000

class ObjectDetectionRequest(BaseModel):
    image: str  # base64 encoded image
    confidence_threshold: float = 0.2
    nms_threshold: float = 0.4

class ImageSegmentationRequest(BaseModel):
    image: str  # base64 encoded image
    confidence_threshold: float = 0.2
    use_yolo: bool = True  # Whether to use YOLO to enhance segmentation
    yolo_confidence: float = 0.2  # YOLO detection confidence threshold

class PoseDetectionRequest(BaseModel):
    image: str  # base64 encoded image
    confidence_threshold: float = 0.2

@app.post("/detect-edges")
async def api_detect_edges(request: EdgeDetectionRequest):
    # Decode the image
    start_time = time.time()
    image = decode_image(request.image)
    
    # Process the image
    edges = detect_edges(
        image, 
        request.low_threshold, 
        request.high_threshold, 
        request.aperture_size
    )
    
    # Encode the result
    result_image = encode_image(edges)
    
    # Calculate processing time
    processing_time = (time.time() - start_time) * 1000  # Convert to ms
    
    return {
        "processed_image": result_image,
        "processing_time_ms": processing_time
    }

@app.post("/extract-features")
async def api_extract_features(request: FeatureExtractionRequest):
    # Decode the image
    start_time = time.time()
    image = decode_image(request.image)
    
    # Extract features
    result_image = extract_features(
        image,
        method=request.method,
        max_features=request.max_features
    )
    
    # Encode the result
    result_image_b64 = encode_image(result_image)
    
    # Calculate processing time
    processing_time = (time.time() - start_time) * 1000  # Convert to ms
    
    return {
        "processed_image": result_image_b64,
        "processing_time_ms": processing_time
    }

@app.post("/detect-objects")
async def api_detect_objects(request: ObjectDetectionRequest):
    # Decode the image
    start_time = time.time()
    image = decode_image(request.image)
    
    # Detect objects
    try:
        result_image, objects = detect_objects(
            image,
            confidence_threshold=request.confidence_threshold,
            nms_threshold=request.nms_threshold
        )
        
        # Encode the result
        result_image_b64 = encode_image(result_image)
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return {
            "processed_image": result_image_b64,
            "objects": objects,
            "processing_time_ms": processing_time
        }
    except FileNotFoundError as e:
        return {
            "error": str(e),
            "message": "Please run python download_yolo_weights.py to download the required model files."
        }

@app.post("/segment-image")
async def api_segment_image(request: ImageSegmentationRequest):
    """
    Enhanced image segmentation using SAM2 (Segment Anything Model 2)
    """
    # Decode the image
    start_time = time.time()
    print(f"Segment image request received, confidence threshold: {request.confidence_threshold}")
    
    try:
        image = decode_image(request.image)
        print(f"Image decoded successfully, shape: {image.shape}")
        
        # Run YOLO object detection if requested
        yolo_results = None
        if request.use_yolo:
            print(f"Running YOLO object detection with confidence threshold: {request.yolo_confidence}")
            try:
                _, yolo_results = detect_objects(
                    image,
                    confidence_threshold=request.yolo_confidence,
                    nms_threshold=0.4
                )
                print(f"YOLO detected {len(yolo_results)} objects")
            except Exception as e:
                print(f"YOLO detection failed, continuing without it: {str(e)}")
                # Continue without YOLO results
        
        # Apply improved segmentation with YOLO guidance
        result_image, class_counts = segment_image(
            image,
            confidence_threshold=request.confidence_threshold,
            yolo_results=yolo_results
        )
        print(f"Segmentation completed, found {len(class_counts)} classes")
        
        # Encode the result
        result_image_b64 = encode_image(result_image)
        print(f"Result image encoded, length: {len(result_image_b64)}")
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Prepare response
        response = {
            "processed_image": result_image_b64,
            "class_counts": class_counts,
            "processing_time_ms": processing_time,
            "model": "SAM2 (Segment Anything Model 2)",
            "yolo_enhanced": request.use_yolo and yolo_results is not None,
            "detected_objects": yolo_results if yolo_results else []
        }
        
        # Validate response before returning
        if not isinstance(result_image_b64, str):
            print(f"WARNING: processed_image is not a string, it's a {type(result_image_b64)}")
            raise ValueError("Processed image is not a valid string")
        
        for class_name, data in class_counts.items():
            if not isinstance(data["pixel_count"], int):
                print(f"WARNING: pixel_count for {class_name} is not an int, it's a {type(data['pixel_count'])}")
                # Convert to int to ensure JSON serialization
                data["pixel_count"] = int(data["pixel_count"])
            
            if not isinstance(data["percentage"], float):
                print(f"WARNING: percentage for {class_name} is not a float, it's a {type(data['percentage'])}")
                # Convert to float to ensure JSON serialization
                data["percentage"] = float(data["percentage"])
        
        print(f"Returning response with {len(response['processed_image'])} bytes of image data")
        return response
    except FileNotFoundError as e:
        error_msg = f"Model file not found: {str(e)}"
        print(f"ERROR: {error_msg}")
        from fastapi import HTTPException
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Model not found",
                "message": "Model initialization failed. Please check your PyTorch installation."
            }
        )
    except Exception as e:
        error_msg = f"Failed to perform enhanced image segmentation: {str(e)}"
        print(f"ERROR: {error_msg}")
        import traceback
        traceback.print_exc()
        from fastapi import HTTPException
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Segmentation failed",
                "message": error_msg
            }
        )

@app.post("/detect-pose")
async def api_detect_pose(request: PoseDetectionRequest):
    """
    Human pose detection using TensorFlow model
    """
    # Decode the image
    start_time = time.time()
    print(f"Pose detection request received, confidence threshold: {request.confidence_threshold}")
    
    try:
        image = decode_image(request.image)
        print(f"Image decoded successfully, shape: {image.shape}")
        
        # Apply pose detection
        result_image, data = detect_human_pose(
            image,
            confidence_threshold=request.confidence_threshold
        )
        print("Pose detection completed")
        
        # Encode the result
        result_image_b64 = encode_image(result_image)
        print(f"Result image encoded, length: {len(result_image_b64)}")
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Prepare response
        response = {
            "processed_image": result_image_b64,
            "processing_time_ms": processing_time,
            "model": "TensorFlow OpenPose"
        }
        
        print(f"Returning response with {len(response['processed_image'])} bytes of image data")
        return response
    except Exception as e:
        error_msg = f"Failed to perform pose detection: {str(e)}"
        print(f"ERROR: {error_msg}")
        import traceback
        traceback.print_exc()
        from fastapi import HTTPException
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Pose detection failed",
                "message": error_msg
            }
        )

@app.post("/benchmark")
async def api_benchmark():
    """
    Run performance benchmarks and return metrics
    """
    print("Benchmark request received")
    
    try:
        # Run benchmarks
        start_time = time.time()
        results = run_benchmarks()
        
        # Calculate total processing time
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        print(f"Benchmarks completed in {processing_time:.2f}ms")
        
        # Return results
        return results
    except Exception as e:
        error_msg = f"Failed to run benchmarks: {str(e)}"
        print(f"ERROR: {error_msg}")
        import traceback
        traceback.print_exc()
        from fastapi import HTTPException
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Benchmark failed",
                "message": error_msg
            }
        )

@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
