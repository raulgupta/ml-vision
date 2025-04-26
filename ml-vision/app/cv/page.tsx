'use client';

export default function ComputerVisionPage() {
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
              <h2 className="text-2xl font-mono text-white/40">COMPUTER VISION LAB</h2>
              <div className="px-4 py-1.5 md:px-6 md:py-2 bg-white/[0.02] backdrop-blur-sm border border-white/[0.03] rounded-lg">
                <span className="font-venus text-base md:text-lg text-white/40">
                  RESEARCH
                </span>
              </div>
            </div>
            
            {/* CV Techniques Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-8">
              {[
                { name: 'Edge Detection', status: 'Available' },
                { name: 'Feature Extraction', status: 'Available' },
                { name: 'Object Detection', status: 'Available' },
                { name: 'Image Segmentation', status: 'Available' },
                { name: 'Pose Estimation', status: 'Available' },
                { name: 'Neural Radiance Fields', status: 'Render Available' }
              ].map((technique, index) => (
                <div 
                  key={index}
                  className="bg-white/[0.02] backdrop-blur-sm border border-white/[0.03] rounded-lg p-4 hover:bg-white/[0.04] transition-all duration-300"
                >
                  <h3 className="text-white/90 font-mono">{technique.name}</h3>
                  <p className={`text-sm ${
                    technique.status === 'Available' || technique.status === 'Render Available'
                      ? 'text-green-400' 
                      : technique.status === 'In Development' 
                        ? 'text-yellow-400' 
                        : 'text-white/40'
                  }`}>
                    {technique.status}
                  </p>
                </div>
              ))}
            </div>
            
            {/* Main Content */}
            <div className="bg-white/[0.02] backdrop-blur-sm border border-white/[0.03] rounded-lg p-6 md:p-8 shadow-[0_0_15px_rgba(255,255,255,0.02)]">
              <p className="text-white/90 mb-4">
                Our computer vision research lab has successfully implemented several cutting-edge techniques
                for image analysis, including edge detection, feature extraction, object detection with YOLO,
                and image segmentation with SAM2.
              </p>
              <p className="text-white/40 mb-6">
                Select a technique from above to explore demos and benchmarks, or visit the home page
                to try these techniques on generated images.
              </p>
              
              <div className="mt-6 bg-white/[0.03] rounded-lg p-6">
                <h3 className="text-white/90 font-mono mb-4">Recent Implementations</h3>
                
                <div className="mb-4">
                  <h4 className="text-white/80 font-mono">Image Segmentation with SAM2</h4>
                  <p className="text-white/60 text-sm mt-2">
                    We've implemented Segment Anything Model 2 (SAM2) for high-quality image segmentation.
                    The implementation features a pink background for better visualization and uses center-point
                    prompting to identify the main object in the image.
                  </p>
                </div>
                
                <div className="mb-4">
                  <h4 className="text-white/80 font-mono">Pose Detection with TensorFlow</h4>
                  <p className="text-white/60 text-sm mt-2">
                    Our human pose detection system uses a TensorFlow model to identify body keypoints
                    and draw a skeleton overlay. The system can detect 18 different body parts and connect
                    them with lines to visualize the human pose.
                  </p>
                </div>
                
                <div className="mb-4">
                  <h4 className="text-white/80 font-mono">Object Detection with YOLOv3</h4>
                  <p className="text-white/60 text-sm mt-2">
                    Our object detection system uses YOLOv3 to identify and classify objects in images
                    with adjustable confidence thresholds.
                  </p>
                </div>
                
                <div className="mb-4">
                  <h4 className="text-white/80 font-mono">Neural Radiance Fields (NeRF)</h4>
                  <p className="text-white/60 text-sm mt-2">
                    Our NeRF implementation creates detailed 3D models from 2D images, allowing for novel view synthesis. 
                    Check out our <a href="/showcase" className="text-green-400 hover:text-green-300">truck demonstration</a> to 
                    see how we've applied this technology to create interactive 3D visualizations from video footage.
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
