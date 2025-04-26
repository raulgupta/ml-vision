export interface CVProcessingResult {
  processingTime: number;
  outputImage?: ImageData;
  features?: Array<{
    x: number;
    y: number;
    score: number;
    descriptor?: Float32Array;
  }>;
  objects?: Array<{
    label: string;
    confidence: number;
    bbox: [number, number, number, number]; // [x, y, width, height]
  }>;
  segmentationData?: {
    [className: string]: {
      pixel_count: number;
      percentage: number;
    }
  };
  modelInfo?: string; // Information about the model used for processing
  error?: string; // Error message if processing failed
}

export class CVService {
  private static instance: CVService;
  private isInitialized: boolean = false;
  
  constructor() {
    if (CVService.instance) {
      return CVService.instance;
    }
    
    CVService.instance = this;
  }
  
  async initialize(): Promise<void> {
    // Placeholder for WebGPU initialization for CV operations
    this.isInitialized = true;
    return Promise.resolve();
  }
  
  // Basic CV operations
  async detectEdges(
    imageData: ImageData, 
    params: {
      lowThreshold?: number;
      highThreshold?: number;
      apertureSize?: number;
    } = {}
  ): Promise<CVProcessingResult> {
    const startTime = performance.now();
    
    try {
      // Convert ImageData to base64
      const canvas = document.createElement('canvas');
      canvas.width = imageData.width;
      canvas.height = imageData.height;
      const ctx = canvas.getContext('2d')!;
      ctx.putImageData(imageData, 0, 0);
      const base64Image = canvas.toDataURL('image/png').split(',')[1];
      
      // Call our API
      const response = await fetch('/api/cv/edge-detection', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image: base64Image,
          low_threshold: params.lowThreshold ?? 50,
          high_threshold: params.highThreshold ?? 150,
          aperture_size: params.apertureSize ?? 3,
        }),
      });
      
      if (!response.ok) {
        throw new Error('Failed to process image');
      }
      
      const data = await response.json();
      
      // Convert base64 back to ImageData
      const img = new Image();
      await new Promise<void>((resolve, reject) => {
        img.onload = () => resolve();
        img.onerror = () => reject(new Error('Failed to load processed image'));
        img.src = `data:image/png;base64,${data.processed_image}`;
      });
      
      const resultCanvas = document.createElement('canvas');
      resultCanvas.width = img.width;
      resultCanvas.height = img.height;
      const resultCtx = resultCanvas.getContext('2d')!;
      resultCtx.drawImage(img, 0, 0);
      const resultImageData = resultCtx.getImageData(0, 0, img.width, img.height);
      
      const endTime = performance.now();
      
      return {
        processingTime: data.processing_time_ms || (endTime - startTime),
        outputImage: resultImageData,
      };
    } catch (error) {
      console.error('Edge detection error:', error);
      return {
        processingTime: performance.now() - startTime,
        error: 'Failed to process image',
      };
    }
  }
  
  async extractFeatures(
    imageData: ImageData,
    params: {
      method?: 'sift' | 'orb';
      maxFeatures?: number;
    } = {}
  ): Promise<CVProcessingResult> {
    const startTime = performance.now();
    
    try {
      // Convert ImageData to base64
      const canvas = document.createElement('canvas');
      canvas.width = imageData.width;
      canvas.height = imageData.height;
      const ctx = canvas.getContext('2d')!;
      ctx.putImageData(imageData, 0, 0);
      const base64Image = canvas.toDataURL('image/png').split(',')[1];
      
      // Call our API
      const response = await fetch('/api/cv/feature-extraction', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image: base64Image,
          method: params.method || 'sift',
          max_features: params.maxFeatures || 1000,
        }),
      });
      
      if (!response.ok) {
        throw new Error('Failed to process image');
      }
      
      const data = await response.json();
      
      // Convert base64 back to ImageData
      const img = new Image();
      await new Promise<void>((resolve, reject) => {
        img.onload = () => resolve();
        img.onerror = () => reject(new Error('Failed to load processed image'));
        img.src = `data:image/png;base64,${data.processed_image}`;
      });
      
      const resultCanvas = document.createElement('canvas');
      resultCanvas.width = img.width;
      resultCanvas.height = img.height;
      const resultCtx = resultCanvas.getContext('2d')!;
      resultCtx.drawImage(img, 0, 0);
      const resultImageData = resultCtx.getImageData(0, 0, img.width, img.height);
      
      const endTime = performance.now();
      
      return {
        processingTime: data.processing_time_ms || (endTime - startTime),
        outputImage: resultImageData,
      };
    } catch (error) {
      console.error('Feature extraction error:', error);
      return {
        processingTime: performance.now() - startTime,
        error: 'Failed to process image',
      };
    }
  }
  
  async detectObjects(
    imageData: ImageData,
    params: {
      confidenceThreshold?: number;
      nmsThreshold?: number;
    } = {}
  ): Promise<CVProcessingResult> {
    const startTime = performance.now();
    
    try {
      // Convert ImageData to base64
      const canvas = document.createElement('canvas');
      canvas.width = imageData.width;
      canvas.height = imageData.height;
      const ctx = canvas.getContext('2d')!;
      ctx.putImageData(imageData, 0, 0);
      const base64Image = canvas.toDataURL('image/png').split(',')[1];
      
      // Call our API
      const response = await fetch('/api/cv/object-detection', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image: base64Image,
          confidence_threshold: params.confidenceThreshold ?? 0.2,
          nms_threshold: params.nmsThreshold ?? 0.4,
        }),
      });
      
      if (!response.ok) {
        throw new Error('Failed to process image');
      }
      
      const data = await response.json();
      
      // Check if there was an error from the Python service
      if (data.error) {
        throw new Error(data.error + (data.message ? `: ${data.message}` : ''));
      }
      
      // Convert base64 back to ImageData
      const img = new Image();
      await new Promise<void>((resolve, reject) => {
        img.onload = () => resolve();
        img.onerror = () => reject(new Error('Failed to load processed image'));
        img.src = `data:image/png;base64,${data.processed_image}`;
      });
      
      const resultCanvas = document.createElement('canvas');
      resultCanvas.width = img.width;
      resultCanvas.height = img.height;
      const resultCtx = resultCanvas.getContext('2d')!;
      resultCtx.drawImage(img, 0, 0);
      const resultImageData = resultCtx.getImageData(0, 0, img.width, img.height);
      
      return {
        processingTime: data.processing_time_ms,
        outputImage: resultImageData,
        objects: data.objects,
      };
    } catch (error) {
      console.error('Object detection error:', error);
      return {
        processingTime: performance.now() - startTime,
        error: error instanceof Error ? error.message : 'Failed to process image',
      };
    }
  }
  
  async segmentImage(
    imageData: ImageData, 
    params: {
      confidenceThreshold?: number;
      useYolo?: boolean;
      yoloConfidence?: number;
    } = {}
  ): Promise<CVProcessingResult> {
    const startTime = performance.now();
    
    try {
      // Convert ImageData to base64
      const canvas = document.createElement('canvas');
      canvas.width = imageData.width;
      canvas.height = imageData.height;
      const ctx = canvas.getContext('2d')!;
      ctx.putImageData(imageData, 0, 0);
      const base64Image = canvas.toDataURL('image/png').split(',')[1];
      
      // Call our API
      const response = await fetch('/api/cv/image-segmentation', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image: base64Image,
          confidence_threshold: params.confidenceThreshold ?? 0.2,
          use_yolo: params.useYolo ?? true,
          yolo_confidence: params.yoloConfidence ?? 0.2,
        }),
      });
      
      if (!response.ok) {
        throw new Error('Failed to process image');
      }
      
      const data = await response.json();
      
      // Check if there was an error from the Python service
      if (data.error) {
        throw new Error(data.error + (data.message ? `: ${data.message}` : ''));
      }
      
      // Convert base64 back to ImageData
      const img = new Image();
      await new Promise<void>((resolve, reject) => {
        img.onload = () => resolve();
        img.onerror = () => reject(new Error('Failed to load processed image'));
        img.src = `data:image/png;base64,${data.processed_image}`;
      });
      
      const resultCanvas = document.createElement('canvas');
      resultCanvas.width = img.width;
      resultCanvas.height = img.height;
      const resultCtx = resultCanvas.getContext('2d')!;
      resultCtx.drawImage(img, 0, 0);
      const resultImageData = resultCtx.getImageData(0, 0, img.width, img.height);
      
      return {
        processingTime: data.processing_time_ms,
        outputImage: resultImageData,
        segmentationData: data.class_counts,
        modelInfo: data.model, // Additional info about the model used
      };
    } catch (error) {
      console.error('Image segmentation error:', error);
      return {
        processingTime: performance.now() - startTime,
        error: error instanceof Error ? error.message : 'Failed to process image',
      };
    }
  }
  
  async detectPose(
    imageData: ImageData,
    params: {
      confidenceThreshold?: number;
    } = {}
  ): Promise<CVProcessingResult> {
    const startTime = performance.now();
    
    try {
      // Convert ImageData to base64
      const canvas = document.createElement('canvas');
      canvas.width = imageData.width;
      canvas.height = imageData.height;
      const ctx = canvas.getContext('2d')!;
      ctx.putImageData(imageData, 0, 0);
      const base64Image = canvas.toDataURL('image/png').split(',')[1];
      
      // Call our API
      const response = await fetch('/api/cv/pose-detection', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image: base64Image,
          confidence_threshold: params.confidenceThreshold ?? 0.2,
        }),
      });
      
      if (!response.ok) {
        throw new Error('Failed to process image');
      }
      
      const data = await response.json();
      
      // Check if there was an error from the Python service
      if (data.error) {
        throw new Error(data.error + (data.message ? `: ${data.message}` : ''));
      }
      
      // Convert base64 back to ImageData
      const img = new Image();
      await new Promise<void>((resolve, reject) => {
        img.onload = () => resolve();
        img.onerror = () => reject(new Error('Failed to load processed image'));
        img.src = `data:image/png;base64,${data.processed_image}`;
      });
      
      const resultCanvas = document.createElement('canvas');
      resultCanvas.width = img.width;
      resultCanvas.height = img.height;
      const resultCtx = resultCanvas.getContext('2d')!;
      resultCtx.drawImage(img, 0, 0);
      const resultImageData = resultCtx.getImageData(0, 0, img.width, img.height);
      
      return {
        processingTime: data.processing_time_ms,
        outputImage: resultImageData,
        modelInfo: data.model, // Additional info about the model used
      };
    } catch (error) {
      console.error('Pose detection error:', error);
      return {
        processingTime: performance.now() - startTime,
        error: error instanceof Error ? error.message : 'Failed to process image',
      };
    }
  }
  
  // This bridges to the NeRF functionality
  async prepareImagesForNeRF(images: ImageData[]): Promise<{
    features: CVProcessingResult[];
    ready: boolean;
  }> {
    // Extract features from all images for NeRF processing
    const results = await Promise.all(
      images.map(img => this.extractFeatures(img))
    );
    
    return {
      features: results,
      ready: results.every(r => (r.features?.length || 0) > 10) // Need enough features
    };
  }
}

// Export singleton
export const cvService = new CVService();
