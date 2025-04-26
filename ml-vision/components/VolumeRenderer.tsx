'use client';

import { useState, useEffect, useRef, Suspense } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Environment, useProgress, Html, useTexture } from '@react-three/drei';
import * as THREE from 'three';

// Model type definition
export type ModelType = 'truck';

// Model configuration
interface ModelConfig {
  path: string;
  totalSlices: number;
  resolution: { width: number; height: number };
  available: boolean;
}

// Model configurations
const MODEL_CONFIGS: Record<ModelType, ModelConfig> = {
  truck: {
    path: '/truck_rgba_slices',
    totalSlices: 288,
    resolution: { width: 352, height: 176 },
    available: true
  }
};

// Loading indicator component
function Loader() {
  const { progress } = useProgress();
  return (
    <Html center>
      <div className="flex flex-col items-center justify-center">
        <div className="text-xl font-venus text-white/90 mb-2">{progress.toFixed(0)}%</div>
        <div className="text-sm text-white/40">Loading...</div>
      </div>
    </Html>
  );
}

// Coming Soon component for unavailable models
function ComingSoon() {
  return (
    <Html center>
      <div className="flex flex-col items-center justify-center">
        <div className="text-xl font-venus text-white/90 mb-2">Coming Soon</div>
        <div className="text-sm text-white/40">This model is currently in development.</div>
      </div>
    </Html>
  );
}

// Volume Model Visualization
interface VolumeModelProps {
  modelType: ModelType;
  densityThreshold: number;
  opacity: number;
}

const VolumeModel: React.FC<VolumeModelProps> = ({ modelType, densityThreshold, opacity }) => {
  const config = MODEL_CONFIGS[modelType];
  
  // If the model is not available, show coming soon
  if (!config.available) {
    return <ComingSoon />;
  }
  
  // Create a group of slices
  const slicesRef = useRef<THREE.Group>(null);
  const [slices, setSlices] = useState<JSX.Element[]>([]);
  const [loading, setLoading] = useState(true);
  
  // Create slices on mount
  useEffect(() => {
    const sliceElements: JSX.Element[] = [];
    const totalSlices = config.totalSlices;
    const spacing = 0.002; // Extremely tight spacing for solid volume with no gaps
    
    // Preload textures in batches to avoid overwhelming the browser
    const loadTextures = async () => {
      const textureLoader = new THREE.TextureLoader();
      
      // Configure texture loader for better quality
      textureLoader.crossOrigin = '';
      
      const loadTexture = (index: number) => {
        return new Promise<THREE.Texture>((resolve, reject) => {
          const sliceNum = index.toString().padStart(4, '0');
          const url = `${config.path}/${sliceNum}_${config.resolution.width}x${config.resolution.height}.png`;
          
          console.log(`[${modelType}] Loading texture: ${url}`);
          
          textureLoader.load(
            url,
            (texture) => {
              // Ensure proper texture settings for clear rendering
              texture.minFilter = THREE.LinearFilter;
              texture.magFilter = THREE.LinearFilter;
              texture.format = THREE.RGBAFormat;
              texture.needsUpdate = true;
              console.log(`[${modelType}] Successfully loaded texture: ${url}`);
              resolve(texture);
            },
            (progressEvent) => {
              // Progress callback
              if (index % 50 === 0) {
                console.log(`[${modelType}] Loading progress for slice ${index}: ${progressEvent.loaded} / ${progressEvent.total}`);
              }
            },
            (error) => {
              // On error, log details and create a placeholder texture
              console.error(`[${modelType}] Error loading texture ${url}:`, error);
              const canvas = document.createElement('canvas');
              canvas.width = config.resolution.width;
              canvas.height = config.resolution.height;
              const ctx = canvas.getContext('2d');
              if (ctx) {
                // Create a visible placeholder with error indication
                ctx.fillStyle = 'rgba(255, 0, 0, 0.2)';
                ctx.fillRect(0, 0, canvas.width, canvas.height);
                ctx.fillStyle = 'white';
                ctx.font = '20px Arial';
                ctx.fillText(`Error: Slice ${index}`, 20, canvas.height / 2);
              }
              const texture = new THREE.CanvasTexture(canvas);
              resolve(texture);
            }
          );
        });
      };
      
      console.log(`Loading ${modelType} model with ${totalSlices} slices`);
      
      // First load a sparse set of slices for quick preview
      const loadSlicesInStages = async () => {
        // Stage 1: Load every 8th slice for quick preview
        for (let i = 0; i < totalSlices; i += 8) {
          try {
            const texture = await loadTexture(i);
            const z = (i - totalSlices / 2) * spacing;
            
            sliceElements.push(
              <mesh key={i} position={[0, 0, z]}>
                <planeGeometry args={[0.8, 0.8]} />
                <meshBasicMaterial 
                  transparent={true}
                  opacity={opacity}
                  alphaTest={densityThreshold}
                  map={texture}
                  side={THREE.DoubleSide}
                  depthWrite={false}
                />
              </mesh>
            );
            
            // Update state to show progress
            if (i % 24 === 0 || i === Math.floor(totalSlices / 8) * 8 - 8) {
              setSlices([...sliceElements]);
            }
          } catch (error) {
            console.error(`Error loading slice ${i}:`, error);
          }
        }
        
        // Stage 2: Load every 4th slice that wasn't loaded in stage 1
        for (let i = 4; i < totalSlices; i += 8) {
          try {
            const texture = await loadTexture(i);
            const z = (i - totalSlices / 2) * spacing;
            
            sliceElements.push(
              <mesh key={i} position={[0, 0, z]}>
                <planeGeometry args={[0.8, 0.8]} />
                <meshBasicMaterial 
                  transparent={true}
                  opacity={opacity}
                  alphaTest={densityThreshold}
                  map={texture}
                  side={THREE.DoubleSide}
                  depthWrite={false}
                />
              </mesh>
            );
            
            // Update state less frequently
            if (i % 32 === 4) {
              setSlices([...sliceElements]);
            }
          } catch (error) {
            console.error(`Error loading slice ${i}:`, error);
          }
        }
        
        // Stage 3: Load all remaining slices
        for (let i = 0; i < totalSlices; i++) {
          // Skip slices we've already loaded
          if (i % 4 === 0) continue;
          
          try {
            const texture = await loadTexture(i);
            const z = (i - totalSlices / 2) * spacing;
            
            sliceElements.push(
              <mesh key={i} position={[0, 0, z]}>
                <planeGeometry args={[0.8, 0.8]} />
                <meshBasicMaterial 
                  transparent={true}
                  opacity={opacity}
                  alphaTest={densityThreshold}
                  map={texture}
                  side={THREE.DoubleSide}
                  depthWrite={false}
                />
              </mesh>
            );
            
            // Update state less frequently for better performance
            if (i % 20 === 2 || i === totalSlices - 1) {
              setSlices([...sliceElements]);
            }
          } catch (error) {
            console.error(`Error loading slice ${i}:`, error);
          }
        }
        
        console.log(`Finished loading all ${totalSlices} slices for ${modelType}`);
        setLoading(false);
      };
      
      loadSlicesInStages();
    };
    
    loadTextures();
  }, [modelType, densityThreshold, opacity, config]);
  
  // Rotate the model with controlled animation
  useFrame((state) => {
    if (slicesRef.current) {
      // Slower rotation to better see the details
      slicesRef.current.rotation.y = state.clock.getElapsedTime() * 0.2;
    }
  });
  
  return (
    <>
      {loading && <Loader />}
      {/* Main volume visualization */}
      <group ref={slicesRef}>
        {slices}
      </group>
      
      {/* Simple bounding box to show volume boundaries */}
      <mesh>
        <boxGeometry args={[0.9, 0.9, 0.9]} />
        <meshBasicMaterial 
          color="#ffffff" 
          wireframe={true} 
          transparent={true} 
          opacity={0.1}
        />
      </mesh>
    </>
  );
};

// Main Volume Renderer Component with Canvas
interface VolumeRendererProps {
  modelType?: ModelType;
  densityThreshold: number;
  stepSize: number; // Not used in simplified version but kept for API compatibility
  opacity: number;
}

export default function VolumeRenderer({ 
  modelType = 'truck', 
  densityThreshold, 
  stepSize, 
  opacity 
}: VolumeRendererProps) {
  return (
    <div className="relative aspect-square w-full max-w-[1024px] mx-auto overflow-hidden rounded-lg border border-white/10">
      <Canvas camera={{ position: [0, 0, 1.5], fov: 45 }}>
        <color attach="background" args={['#918868']} />
        <ambientLight intensity={0.8} />
        <directionalLight position={[1, 1, 1]} intensity={0.5} />
        <Suspense fallback={<Loader />}>
          <VolumeModel 
            modelType={modelType}
            densityThreshold={densityThreshold}
            opacity={opacity}
          />
          <OrbitControls enableDamping dampingFactor={0.05} />
          <Environment preset="city" />
        </Suspense>
      </Canvas>
    </div>
  );
}
