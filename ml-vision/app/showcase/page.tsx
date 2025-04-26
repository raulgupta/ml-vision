'use client';

import { useState, useEffect } from 'react';
import dynamic from 'next/dynamic';
import type { ModelType } from '../../components/VolumeRenderer';

// Model information
interface ModelInfo {
  name: string;
  description: string;
  status: 'available' | 'coming-soon';
  sliceCount: number;
  resolution: string;
  videoPath?: string;
  densityMapPath?: string;
}

// Model information
const MODELS: Record<ModelType, ModelInfo> = {
  truck: {
    name: 'Truck',
    description: 'A detailed 3D truck model rendered using Neural Radiance Fields technology.',
    status: 'available',
    sliceCount: 288,
    resolution: '352×176',
    videoPath: '/truck.mp4',
    densityMapPath: '/truck_density.png'
  }
};

// Loading fallback component
const LoadingFallback = () => (
  <div className="aspect-square w-full max-w-[1024px] mx-auto flex items-center justify-center bg-white/[0.01] rounded-lg border border-white/10">
    <div className="text-center">
      <div className="text-xl font-venus text-white/90 mb-2">Loading 3D Renderer</div>
      <div className="text-sm text-white/40">Please wait while the volumetric renderer initializes...</div>
    </div>
  </div>
);

// Import the VolumeRenderer component with dynamic import to prevent SSR
const VolumeRenderer = dynamic(
  () => import('../../components/VolumeRenderer'),
  { 
    ssr: false,
    loading: () => <LoadingFallback />
  }
);

// Model selector component
interface ModelSelectorProps {
  selectedModel: ModelType;
  onSelectModel: (model: ModelType) => void;
}

const ModelSelector: React.FC<ModelSelectorProps> = ({ selectedModel, onSelectModel }) => {
  return (
    <div className="flex flex-wrap gap-3 mb-6">
      {(Object.keys(MODELS) as ModelType[]).map((modelType) => {
        const model = MODELS[modelType];
        const isSelected = modelType === selectedModel;
        const isAvailable = model.status === 'available';
        
        return (
          <button
            key={modelType}
            onClick={() => isAvailable && onSelectModel(modelType)}
            className={`
              px-4 py-2 rounded-lg font-venus text-base transition-all duration-300
              ${isSelected 
                ? 'bg-white/[0.08] text-white/90 border border-white/[0.1]' 
                : 'bg-white/[0.02] text-white/40 border border-white/[0.03]'}
              ${isAvailable 
                ? 'hover:bg-white/[0.04] hover:text-white/60 cursor-pointer' 
                : 'opacity-50 cursor-not-allowed'}
            `}
          >
            {model.name}
            {!isAvailable && (
              <span className="ml-2 text-xs bg-white/10 px-2 py-0.5 rounded">Soon</span>
            )}
          </button>
        );
      })}
    </div>
  );
};

// Main Showcase Page
export default function ShowcasePage() {
  const [selectedModel, setSelectedModel] = useState<ModelType>('truck');
  const [densityThreshold, setDensityThreshold] = useState(0.01); // Lower default threshold for better visibility
  const [stepSize, setStepSize] = useState(0.01);
  const [opacity, setOpacity] = useState(0.8); // Slightly lower opacity for better transparency
  const [gpuInfo, setGpuInfo] = useState<{ supported: boolean; info?: string }>({ supported: false });
  
  const currentModel = MODELS[selectedModel];

  // Check WebGPU support
  useEffect(() => {
    const checkGPU = async () => {
      try {
        if (typeof window === 'undefined') {
          return { supported: false, info: 'Not in browser environment' };
        }
        
        if (!('gpu' in navigator)) {
          return { supported: false, info: 'WebGPU not supported' };
        }
        
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) {
          return { supported: false, info: 'No WebGPU adapter found' };
        }
        
        setGpuInfo({ 
          supported: true, 
          info: `WebGPU supported: ${adapter.name || 'Unknown GPU'}` 
        });
      } catch (error) {
        setGpuInfo({ 
          supported: false, 
          info: error instanceof Error ? error.message : 'Unknown error' 
        });
      }
    };
    
    checkGPU();
  }, []);

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
            <div className="flex justify-between items-center mb-12">
              <h2 className="text-2xl font-mono text-white/40">NERF SHOWCASE</h2>
              <div className="px-4 py-1.5 md:px-6 md:py-2 bg-white/[0.02] backdrop-blur-sm border border-white/[0.03] rounded-lg">
                <span className="font-venus text-base md:text-lg text-white/40">
                  EXPERIMENTAL
                </span>
              </div>
            </div>
            
            {/* Model Selector */}
            <div className="bg-white/[0.02] backdrop-blur-sm border border-white/[0.03] rounded-lg p-6 md:p-8 shadow-[0_0_15px_rgba(255,255,255,0.02)] mb-8">
              <h3 className="text-sm font-mono text-white/40 mb-4">SELECT MODEL</h3>
              <ModelSelector 
                selectedModel={selectedModel}
                onSelectModel={setSelectedModel}
              />
              <p className="text-white/60 text-sm">
                {currentModel.description}
              </p>
            </div>
            
            {/* GPU Status */}
            <div className="bg-white/[0.02] backdrop-blur-sm border border-white/[0.03] rounded-lg p-6 md:p-8 shadow-[0_0_15px_rgba(255,255,255,0.02)] mb-8">
              <h3 className="text-sm font-mono text-white/40 mb-4">GPU STATUS</h3>
              <div className="flex items-baseline gap-2">
                <span className={`text-xl font-venus ${gpuInfo.supported ? 'text-green-400' : 'text-red-400'}`}>
                  {gpuInfo.supported ? 'SUPPORTED' : 'NOT SUPPORTED'}
                </span>
              </div>
              {gpuInfo.info && (
                <p className="text-white/60 text-sm mt-2">{gpuInfo.info}</p>
              )}
            </div>
            
            {/* 3D Viewer */}
            <div className="bg-white/[0.02] backdrop-blur-sm border border-white/[0.03] rounded-lg p-6 md:p-8 shadow-[0_0_15px_rgba(255,255,255,0.02)] mb-8">
              <h3 className="text-sm font-mono text-white/40 mb-4">VOLUMETRIC RENDERING</h3>
              <VolumeRenderer 
                modelType={selectedModel}
                densityThreshold={densityThreshold}
                stepSize={stepSize}
                opacity={opacity}
              />
            </div>
            
            {/* Controls */}
            <div className="bg-white/[0.02] backdrop-blur-sm border border-white/[0.03] rounded-lg p-6 md:p-8 shadow-[0_0_15px_rgba(255,255,255,0.02)] mb-8">
              <h3 className="text-sm font-mono text-white/40 mb-4">RENDERING PARAMETERS</h3>
              
              {/* Density Threshold */}
              <div className="mb-4">
                <div className="flex justify-between items-center mb-2">
                  <label className="text-white/60 text-sm">Density Threshold</label>
                  <span className="text-white/90 font-venus">{densityThreshold.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.01"
                  value={densityThreshold}
                  onChange={(e) => setDensityThreshold(parseFloat(e.target.value))}
                  className="w-full"
                />
                <p className="text-white/40 text-xs mt-1">
                  Controls which parts of the volume are visible based on density
                </p>
              </div>
              
              {/* Step Size */}
              <div className="mb-4">
                <div className="flex justify-between items-center mb-2">
                  <label className="text-white/60 text-sm">Step Size</label>
                  <span className="text-white/90 font-venus">{stepSize.toFixed(3)}</span>
                </div>
                <input
                  type="range"
                  min="0.001"
                  max="0.05"
                  step="0.001"
                  value={stepSize}
                  onChange={(e) => setStepSize(parseFloat(e.target.value))}
                  className="w-full"
                />
                <p className="text-white/40 text-xs mt-1">
                  Controls the precision of the ray marching algorithm
                </p>
              </div>
              
              {/* Opacity */}
              <div>
                <div className="flex justify-between items-center mb-2">
                  <label className="text-white/60 text-sm">Opacity</label>
                  <span className="text-white/90 font-venus">{opacity.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.01"
                  value={opacity}
                  onChange={(e) => setOpacity(parseFloat(e.target.value))}
                  className="w-full"
                />
                <p className="text-white/40 text-xs mt-1">
                  Controls the overall transparency of the volume
                </p>
              </div>
            </div>
            
            {/* Density Map Visualization - Only show for available models */}
            {currentModel.status === 'available' && currentModel.densityMapPath && (
              <div className="bg-white/[0.02] backdrop-blur-sm border border-white/[0.03] rounded-lg p-6 md:p-8 shadow-[0_0_15px_rgba(255,255,255,0.02)] mb-8">
                <h3 className="text-sm font-mono text-white/40 mb-4">DENSITY MAP</h3>
                
                <div className="flex flex-col md:flex-row gap-6 items-center">
                  <div className="w-full md:w-1/2">
                    <div className="relative aspect-square w-full max-w-[400px] mx-auto overflow-hidden rounded-lg border border-white/10">
                      <img 
                        src={currentModel.densityMapPath} 
                        alt={`${currentModel.name} Density Map`} 
                        className="w-full h-full object-cover"
                      />
                    </div>
                  </div>
                  
                  <div className="w-full md:w-1/2 space-y-4">
                    <h4 className="text-white/70 font-mono mb-2">2D Representation</h4>
                    <p className="text-white/60 text-sm">
                      This density map shows a 2D representation of the volumetric data. Each pixel's brightness corresponds 
                      to the density value at that point in the volume. The brighter areas indicate higher density regions 
                      where the {currentModel.name.toLowerCase()} model is present.
                    </p>
                    
                    <div className="mt-6">
                      <div className="bg-black/30 p-3 rounded-lg font-mono text-xs text-white/70 overflow-x-auto">
                        <pre>python /opt/instant-ngp/scripts/colmap2nerf.py --video_in truck.mp4 --video_fps 2 --run_colmap --overwrite</pre>
                      </div>
                      <p className="text-white/40 text-xs mt-2">
                        Command used to generate the NeRF model from the source video
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            )}
            
            {/* Source Video - Only show for available models with video */}
            {currentModel.status === 'available' && currentModel.videoPath && (
              <div className="bg-white/[0.02] backdrop-blur-sm border border-white/[0.03] rounded-lg p-6 md:p-8 shadow-[0_0_15px_rgba(255,255,255,0.02)] mb-8">
                <h3 className="text-sm font-mono text-white/40 mb-4">SOURCE VIDEO</h3>
                
                <div className="flex flex-col md:flex-row gap-6 items-center">
                  <div className="w-full md:w-1/2">
                    <div className="relative aspect-video w-full max-w-[400px] mx-auto overflow-hidden rounded-lg border border-white/10">
                      <video 
                        src={currentModel.videoPath} 
                        controls
                        className="w-full h-full object-cover"
                      />
                    </div>
                  </div>
                  
                  <div className="w-full md:w-1/2 space-y-4">
                    <h4 className="text-white/70 font-mono mb-2">Original Footage</h4>
                    <p className="text-white/60 text-sm">
                      This is the original video footage used to create the Neural Radiance Field (NeRF) model. 
                      The NeRF algorithm processes multiple views of the {currentModel.name.toLowerCase()} to reconstruct a 3D volumetric representation,
                      allowing for novel view synthesis and interactive exploration of the 3D scene.
                    </p>
                    
                    <div className="mt-6">
                      <p className="text-white/40 text-xs">
                        The video contains multiple angles and perspectives that provide the necessary information for the NeRF algorithm
                        to understand the 3D structure and appearance of the object.
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            )}
            
            {/* Technical Explanation */}
            <div className="bg-white/[0.02] backdrop-blur-sm border border-white/[0.03] rounded-lg p-6 md:p-8 shadow-[0_0_15px_rgba(255,255,255,0.02)]">
              <h3 className="text-sm font-mono text-white/40 mb-4">TECHNICAL DETAILS</h3>
              
              <div className="space-y-4">
                <p className="text-white/90">
                  This showcase demonstrates volumetric rendering of a Neural Radiance Field (NeRF) using WebGL shaders and Three.js.
                </p>
                
                <div>
                  <h4 className="text-white/70 font-mono mb-2">Implementation</h4>
                  <p className="text-white/60 text-sm">
                    The {currentModel.name.toLowerCase()} model is represented as a stack of {currentModel.sliceCount} RGBA texture slices, each with a resolution of {currentModel.resolution} pixels.
                    These slices are loaded into the GPU and rendered using a custom ray marching shader that samples the volume
                    along view rays to create the 3D effect. The downloadable dataset contains all {currentModel.sliceCount} slices used in this visualization.
                  </p>
                </div>
                
                <div>
                  <h4 className="text-white/70 font-mono mb-2">Ray Marching</h4>
                  <p className="text-white/60 text-sm">
                    The rendering technique steps through the volume from front to back, accumulating color and opacity
                    based on the density values stored in the alpha channel of each slice. This creates a realistic 3D appearance
                    from 2D image slices. The density map image provides a 2D visualization of this volumetric data.
                  </p>
                </div>
                
                <div>
                  <h4 className="text-white/70 font-mono mb-2">Performance</h4>
                  <p className="text-white/60 text-sm">
                    The renderer uses early ray termination to improve performance by stopping the ray march when
                    sufficient opacity has accumulated. WebGPU acceleration is used when available to further enhance performance.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </main>
  );
}
