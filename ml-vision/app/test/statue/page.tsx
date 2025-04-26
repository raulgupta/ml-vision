'use client';

import { useState, useEffect, useRef, Suspense } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Environment, Html } from '@react-three/drei';
import * as THREE from 'three';

// Statue configuration
const STATUE_CONFIG = {
  path: '/statue_rgba_slices',
  totalSlices: 224,
  resolution: { width: 256, height: 224 },
};

// Loading indicator component
function Loader({ progress }: { progress: number }) {
  return (
    <Html center>
      <div className="flex flex-col items-center justify-center">
        <div className="text-xl font-mono text-white/90 mb-2">{progress.toFixed(0)}%</div>
        <div className="text-sm text-white/40">Loading...</div>
      </div>
    </Html>
  );
}

// Debug info component
function DebugInfo({ 
  loadedSlices, 
  totalSlices, 
  errors, 
  boxSize, 
  planeSize, 
  spacing,
  densityThreshold,
  opacity
}: { 
  loadedSlices: number; 
  totalSlices: number; 
  errors: number;
  boxSize: [number, number, number];
  planeSize: [number, number];
  spacing: number;
  densityThreshold: number;
  opacity: number;
}) {
  return (
    <Html position={[1.5, 0, 0]}>
      <div className="bg-black/80 p-4 rounded-lg text-white text-xs font-mono" style={{ width: '200px' }}>
        <h3 className="text-sm font-bold mb-2">Debug Info</h3>
        <div className="space-y-1">
          <div>Loaded: {loadedSlices}/{totalSlices} slices</div>
          <div>Errors: {errors}</div>
          <div>Box: [{boxSize.join(', ')}]</div>
          <div>Plane: [{planeSize.join(', ')}]</div>
          <div>Spacing: {spacing.toFixed(4)}</div>
          <div>Threshold: {densityThreshold.toFixed(4)}</div>
          <div>Opacity: {opacity.toFixed(2)}</div>
        </div>
      </div>
    </Html>
  );
}

// Coordinate axes with labels
function CoordinateAxes({ size = 1 }) {
  return (
    <>
      {/* X Axis - Red */}
      <group>
        <mesh position={[size/2, 0, 0]}>
          <boxGeometry args={[size, 0.01, 0.01]} />
          <meshBasicMaterial color="red" />
        </mesh>
        <Html position={[size + 0.05, 0, 0]}>
          <div className="text-red-500 font-bold">X</div>
        </Html>
      </group>
      
      {/* Y Axis - Green */}
      <group>
        <mesh position={[0, size/2, 0]}>
          <boxGeometry args={[0.01, size, 0.01]} />
          <meshBasicMaterial color="green" />
        </mesh>
        <Html position={[0, size + 0.05, 0]}>
          <div className="text-green-500 font-bold">Y</div>
        </Html>
      </group>
      
      {/* Z Axis - Blue */}
      <group>
        <mesh position={[0, 0, size/2]}>
          <boxGeometry args={[0.01, 0.01, size]} />
          <meshBasicMaterial color="blue" />
        </mesh>
        <Html position={[0, 0, size + 0.05]}>
          <div className="text-blue-500 font-bold">Z</div>
        </Html>
      </group>
    </>
  );
}

// Statue Model Visualization
function StatueModel({ 
  boxSize, 
  planeSize, 
  spacing, 
  densityThreshold, 
  opacity,
  color
}: { 
  boxSize: [number, number, number]; 
  planeSize: [number, number]; 
  spacing: number;
  densityThreshold: number;
  opacity: number;
  color: THREE.Color;
}) {
  // Create a group of slices
  const slicesRef = useRef<THREE.Group>(null);
  const [slices, setSlices] = useState<JSX.Element[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadedCount, setLoadedCount] = useState(0);
  const [errorCount, setErrorCount] = useState(0);
  const [loadingProgress, setLoadingProgress] = useState(0);
  
  // Create slices on mount
  useEffect(() => {
    const sliceElements: JSX.Element[] = [];
    const totalSlices = STATUE_CONFIG.totalSlices;
    
    // Reset counters
    setLoadedCount(0);
    setErrorCount(0);
    setLoadingProgress(0);
    
    // Preload textures in batches to avoid overwhelming the browser
    const loadTextures = async () => {
      const textureLoader = new THREE.TextureLoader();
      
      // Configure texture loader for better quality
      textureLoader.crossOrigin = '';
      
      const loadTexture = (index: number) => {
        return new Promise<THREE.Texture>((resolve, reject) => {
          const sliceNum = index.toString().padStart(4, '0');
          const url = `${STATUE_CONFIG.path}/${sliceNum}_${STATUE_CONFIG.resolution.width}x${STATUE_CONFIG.resolution.height}.png`;
          
          console.log(`Loading texture: ${url}`);
          
          textureLoader.load(
            url,
            (texture) => {
              // Ensure proper texture settings for clear rendering
              texture.minFilter = THREE.LinearFilter;
              texture.magFilter = THREE.LinearFilter;
              texture.format = THREE.RGBAFormat;
              texture.needsUpdate = true;
              console.log(`Successfully loaded texture: ${url}`);
              setLoadedCount(prev => prev + 1);
              resolve(texture);
            },
            (progressEvent) => {
              // Progress callback
              if (index % 10 === 0) {
                console.log(`Loading progress for slice ${index}: ${progressEvent.loaded} / ${progressEvent.total}`);
              }
            },
            (error) => {
              // On error, log details and create a placeholder texture
              console.error(`Error loading texture ${url}:`, error);
              setErrorCount(prev => prev + 1);
              
              const canvas = document.createElement('canvas');
              canvas.width = STATUE_CONFIG.resolution.width;
              canvas.height = STATUE_CONFIG.resolution.height;
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
      
      console.log(`Loading statue model with ${totalSlices} slices`);
      
      // Load all slices sequentially for simplicity in the test page
      for (let i = 0; i < totalSlices; i++) {
        try {
          const texture = await loadTexture(i);
          
          // Calculate position - center the volume
          const z = (i - totalSlices / 2) * spacing;
          
          // Create slice with simple material to show actual data
          sliceElements.push(
            <mesh key={i} position={[0, 0, z]}>
              <planeGeometry args={planeSize} />
              <meshBasicMaterial 
                transparent={true}
                opacity={opacity}
                alphaTest={densityThreshold}
                map={texture}
                side={THREE.DoubleSide}
                depthWrite={false}
                color={color}
              />
            </mesh>
          );
          
          // Update state every few slices to show loading progress
          if (i % 5 === 0 || i === totalSlices - 1) {
            setSlices([...sliceElements]);
            setLoadingProgress((i + 1) / totalSlices * 100);
          }
        } catch (error) {
          console.error(`Error loading slice ${i}:`, error);
        }
      }
      
      console.log(`Finished loading all ${totalSlices} slices for statue`);
      setLoading(false);
    };
    
    loadTextures();
  }, [boxSize, planeSize, spacing, densityThreshold, opacity, color]);
  
  // Rotate the model with controlled animation
  useFrame((state) => {
    if (slicesRef.current) {
      // Slower rotation to better see the details
      slicesRef.current.rotation.y = state.clock.getElapsedTime() * 0.1;
    }
  });
  
  return (
    <>
      {loading && <Loader progress={loadingProgress} />}
      
      {/* Debug information */}
      <DebugInfo 
        loadedSlices={loadedCount} 
        totalSlices={STATUE_CONFIG.totalSlices} 
        errors={errorCount}
        boxSize={boxSize}
        planeSize={planeSize}
        spacing={spacing}
        densityThreshold={densityThreshold}
        opacity={opacity}
      />
      
      {/* Main volume visualization */}
      <group ref={slicesRef}>
        {slices}
      </group>
      
      {/* Visible bounding box for debugging */}
      <mesh>
        <boxGeometry args={boxSize} />
        <meshBasicMaterial 
          color="#ffffff" 
          wireframe={true} 
          transparent={true} 
          opacity={0.3}
        />
      </mesh>
      
      {/* Coordinate axes */}
      <CoordinateAxes size={1} />
    </>
  );
}

// Main Test Page
export default function StatueTestPage() {
  // Adjustable parameters with UI controls - initialized to match reference image
  const [boxWidth, setBoxWidth] = useState(0.71);
  const [boxHeight, setBoxHeight] = useState(0.71);
  const [boxDepth, setBoxDepth] = useState(0.71);
  
  const [planeWidth, setPlaneWidth] = useState(0.7);
  const [planeHeight, setPlaneHeight] = useState(0.7 * (224/256)); // Adjusted for aspect ratio
  
  // Using values from the reference image
  const [spacing, setSpacing] = useState(0.003);
  const [densityThreshold, setDensityThreshold] = useState(0.006); // From reference image
  const [opacity, setOpacity] = useState(1.0);
  
  // Color for the statue (teal/turquoise from reference)
  const [colorR, setColorR] = useState(0.2);
  const [colorG, setColorG] = useState(0.8);
  const [colorB, setColorB] = useState(0.8);
  
  const [cameraX, setCameraX] = useState(0);
  const [cameraY, setCameraY] = useState(0);
  const [cameraZ, setCameraZ] = useState(2);
  const [fov, setFov] = useState(58.6); // From reference image

  // Computed values
  const boxSize: [number, number, number] = [boxWidth, boxHeight, boxDepth];
  const planeSize: [number, number] = [planeWidth, planeHeight];
  const cameraPosition: [number, number, number] = [cameraX, cameraY, cameraZ];
  const color = new THREE.Color(colorR, colorG, colorB);

  return (
    <main className="min-h-screen w-full relative overflow-hidden bg-gray-900">
      <div className="container mx-auto px-4 py-8">
        <div className="mt-8">
          <h1 className="text-2xl font-mono text-white/80 mb-4">Statue Model Test Page</h1>
          
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* 3D Viewer */}
            <div className="lg:col-span-2">
              <div className="aspect-square w-full bg-black rounded-lg border border-white/10 overflow-hidden">
                <Canvas camera={{ position: cameraPosition, fov: fov }}>
                  <color attach="background" args={['#0a0a0a']} />
                  <ambientLight intensity={0.8} />
                  <directionalLight position={[1, 1, 1]} intensity={0.5} />
                  <Suspense fallback={<Loader progress={0} />}>
                    <StatueModel 
                      boxSize={boxSize}
                      planeSize={planeSize}
                      spacing={spacing}
                      densityThreshold={densityThreshold}
                      opacity={opacity}
                      color={color}
                    />
                    <OrbitControls 
                      enableDamping 
                      dampingFactor={0.05} 
                      rotateSpeed={0.5}
                    />
                    <Environment preset="city" />
                  </Suspense>
                </Canvas>
              </div>
            </div>
            
            {/* Controls */}
            <div className="space-y-6">
              {/* Bounding Box Controls */}
              <div className="bg-white/[0.02] backdrop-blur-sm border border-white/[0.03] rounded-lg p-4 shadow-[0_0_15px_rgba(255,255,255,0.02)]">
                <h3 className="text-sm font-mono text-white/40 mb-3">Bounding Box</h3>
                
                <div className="space-y-3">
                  <div>
                    <div className="flex justify-between items-center mb-1">
                      <label className="text-white/60 text-xs">Width</label>
                      <span className="text-white/90 font-mono text-xs">{boxWidth.toFixed(3)}</span>
                    </div>
                    <input
                      type="range"
                      min="0.1"
                      max="2"
                      step="0.01"
                      value={boxWidth}
                      onChange={(e) => setBoxWidth(parseFloat(e.target.value))}
                      className="w-full"
                    />
                  </div>
                  
                  <div>
                    <div className="flex justify-between items-center mb-1">
                      <label className="text-white/60 text-xs">Height</label>
                      <span className="text-white/90 font-mono text-xs">{boxHeight.toFixed(3)}</span>
                    </div>
                    <input
                      type="range"
                      min="0.1"
                      max="2"
                      step="0.01"
                      value={boxHeight}
                      onChange={(e) => setBoxHeight(parseFloat(e.target.value))}
                      className="w-full"
                    />
                  </div>
                  
                  <div>
                    <div className="flex justify-between items-center mb-1">
                      <label className="text-white/60 text-xs">Depth</label>
                      <span className="text-white/90 font-mono text-xs">{boxDepth.toFixed(3)}</span>
                    </div>
                    <input
                      type="range"
                      min="0.1"
                      max="2"
                      step="0.01"
                      value={boxDepth}
                      onChange={(e) => setBoxDepth(parseFloat(e.target.value))}
                      className="w-full"
                    />
                  </div>
                </div>
              </div>
              
              {/* Plane Size Controls */}
              <div className="bg-white/[0.02] backdrop-blur-sm border border-white/[0.03] rounded-lg p-4 shadow-[0_0_15px_rgba(255,255,255,0.02)]">
                <h3 className="text-sm font-mono text-white/40 mb-3">Plane Size</h3>
                
                <div className="space-y-3">
                  <div>
                    <div className="flex justify-between items-center mb-1">
                      <label className="text-white/60 text-xs">Width</label>
                      <span className="text-white/90 font-mono text-xs">{planeWidth.toFixed(3)}</span>
                    </div>
                    <input
                      type="range"
                      min="0.1"
                      max="2"
                      step="0.01"
                      value={planeWidth}
                      onChange={(e) => setPlaneWidth(parseFloat(e.target.value))}
                      className="w-full"
                    />
                  </div>
                  
                  <div>
                    <div className="flex justify-between items-center mb-1">
                      <label className="text-white/60 text-xs">Height</label>
                      <span className="text-white/90 font-mono text-xs">{planeHeight.toFixed(3)}</span>
                    </div>
                    <input
                      type="range"
                      min="0.1"
                      max="2"
                      step="0.01"
                      value={planeHeight}
                      onChange={(e) => setPlaneHeight(parseFloat(e.target.value))}
                      className="w-full"
                    />
                  </div>
                </div>
              </div>
              
              {/* Rendering Parameters */}
              <div className="bg-white/[0.02] backdrop-blur-sm border border-white/[0.03] rounded-lg p-4 shadow-[0_0_15px_rgba(255,255,255,0.02)]">
                <h3 className="text-sm font-mono text-white/40 mb-3">Rendering</h3>
                
                <div className="space-y-3">
                  <div>
                    <div className="flex justify-between items-center mb-1">
                      <label className="text-white/60 text-xs">Spacing</label>
                      <span className="text-white/90 font-mono text-xs">{spacing.toFixed(4)}</span>
                    </div>
                    <input
                      type="range"
                      min="0.0001"
                      max="0.01"
                      step="0.0001"
                      value={spacing}
                      onChange={(e) => setSpacing(parseFloat(e.target.value))}
                      className="w-full"
                    />
                  </div>
                  
                  <div>
                    <div className="flex justify-between items-center mb-1">
                      <label className="text-white/60 text-xs">Density Threshold</label>
                      <span className="text-white/90 font-mono text-xs">{densityThreshold.toFixed(4)}</span>
                    </div>
                    <input
                      type="range"
                      min="0.001"
                      max="0.05"
                      step="0.001"
                      value={densityThreshold}
                      onChange={(e) => setDensityThreshold(parseFloat(e.target.value))}
                      className="w-full"
                    />
                  </div>
                  
                  <div>
                    <div className="flex justify-between items-center mb-1">
                      <label className="text-white/60 text-xs">Opacity</label>
                      <span className="text-white/90 font-mono text-xs">{opacity.toFixed(2)}</span>
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
                  </div>
                </div>
              </div>
              
              {/* Color Controls */}
              <div className="bg-white/[0.02] backdrop-blur-sm border border-white/[0.03] rounded-lg p-4 shadow-[0_0_15px_rgba(255,255,255,0.02)]">
                <h3 className="text-sm font-mono text-white/40 mb-3">Color</h3>
                
                <div className="space-y-3">
                  <div>
                    <div className="flex justify-between items-center mb-1">
                      <label className="text-white/60 text-xs">Red</label>
                      <span className="text-white/90 font-mono text-xs">{colorR.toFixed(2)}</span>
                    </div>
                    <input
                      type="range"
                      min="0"
                      max="1"
                      step="0.01"
                      value={colorR}
                      onChange={(e) => setColorR(parseFloat(e.target.value))}
                      className="w-full"
                    />
                  </div>
                  
                  <div>
                    <div className="flex justify-between items-center mb-1">
                      <label className="text-white/60 text-xs">Green</label>
                      <span className="text-white/90 font-mono text-xs">{colorG.toFixed(2)}</span>
                    </div>
                    <input
                      type="range"
                      min="0"
                      max="1"
                      step="0.01"
                      value={colorG}
                      onChange={(e) => setColorG(parseFloat(e.target.value))}
                      className="w-full"
                    />
                  </div>
                  
                  <div>
                    <div className="flex justify-between items-center mb-1">
                      <label className="text-white/60 text-xs">Blue</label>
                      <span className="text-white/90 font-mono text-xs">{colorB.toFixed(2)}</span>
                    </div>
                    <input
                      type="range"
                      min="0"
                      max="1"
                      step="0.01"
                      value={colorB}
                      onChange={(e) => setColorB(parseFloat(e.target.value))}
                      className="w-full"
                    />
                  </div>
                  
                  <div className="h-6 rounded-md" style={{ backgroundColor: `rgb(${colorR*255}, ${colorG*255}, ${colorB*255})` }}></div>
                </div>
              </div>
              
              {/* Camera Controls */}
              <div className="bg-white/[0.02] backdrop-blur-sm border border-white/[0.03] rounded-lg p-4 shadow-[0_0_15px_rgba(255,255,255,0.02)]">
                <h3 className="text-sm font-mono text-white/40 mb-3">Camera</h3>
                
                <div className="space-y-3">
                  <div>
                    <div className="flex justify-between items-center mb-1">
                      <label className="text-white/60 text-xs">X Position</label>
                      <span className="text-white/90 font-mono text-xs">{cameraX.toFixed(1)}</span>
                    </div>
                    <input
                      type="range"
                      min="-3"
                      max="3"
                      step="0.1"
                      value={cameraX}
                      onChange={(e) => setCameraX(parseFloat(e.target.value))}
                      className="w-full"
                    />
                  </div>
                  
                  <div>
                    <div className="flex justify-between items-center mb-1">
                      <label className="text-white/60 text-xs">Y Position</label>
                      <span className="text-white/90 font-mono text-xs">{cameraY.toFixed(1)}</span>
                    </div>
                    <input
                      type="range"
                      min="-3"
                      max="3"
                      step="0.1"
                      value={cameraY}
                      onChange={(e) => setCameraY(parseFloat(e.target.value))}
                      className="w-full"
                    />
                  </div>
                  
                  <div>
                    <div className="flex justify-between items-center mb-1">
                      <label className="text-white/60 text-xs">Z Position</label>
                      <span className="text-white/90 font-mono text-xs">{cameraZ.toFixed(1)}</span>
                    </div>
                    <input
                      type="range"
                      min="0.5"
                      max="5"
                      step="0.1"
                      value={cameraZ}
                      onChange={(e) => setCameraZ(parseFloat(e.target.value))}
                      className="w-full"
                    />
                  </div>
                  
                  <div>
                    <div className="flex justify-between items-center mb-1">
                      <label className="text-white/60 text-xs">Field of View</label>
                      <span className="text-white/90 font-mono text-xs">{fov.toFixed(1)}°</span>
                    </div>
                    <input
                      type="range"
                      min="10"
                      max="100"
                      step="0.1"
                      value={fov}
                      onChange={(e) => setFov(parseFloat(e.target.value))}
                      className="w-full"
                    />
                  </div>
                </div>
              </div>
              
              {/* Reset Button */}
              <button
                onClick={() => {
                  // Reset to values from reference image
                  setBoxWidth(0.71);
                  setBoxHeight(0.71);
                  setBoxDepth(0.71);
                  setPlaneWidth(0.7);
                  setPlaneHeight(0.7 * (224/256));
                  setSpacing(0.003);
                  setDensityThreshold(0.006);
                  setOpacity(1.0);
                  setColorR(0.2);
                  setColorG(0.8);
                  setColorB(0.8);
                  setCameraX(0);
                  setCameraY(0);
                  setCameraZ(2);
                  setFov(58.6);
                }}
                className="w-full py-2 bg-white/[0.05] hover:bg-white/[0.1] text-white/60 hover:text-white/90 rounded-lg transition-all duration-200"
              >
                Reset All Parameters
              </button>
            </div>
          </div>
        </div>
      </div>
    </main>
  );
}
