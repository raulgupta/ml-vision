'use client';

import { 
  SiPytorch, 
  SiTensorflow, 
  SiNvidia, 
  SiHuggingface, 
  SiDocker, 
  SiOpencv, 
  SiGithubactions 
} from 'react-icons/si';

export default function ResumePage() {
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
            <div className="mb-12">
              <h2 className="text-2xl font-mono text-white/40">RESUME</h2>
            </div>
            
            {/* Profile Section */}
            <div className="bg-white/[0.02] backdrop-blur-sm border border-white/[0.03] rounded-lg p-6 md:p-8 shadow-[0_0_15px_rgba(255,255,255,0.02)] mb-8">
              <div className="flex flex-col md:flex-row items-start md:items-center gap-6">
                <div className="w-24 h-24 rounded-full bg-white/10 flex items-center justify-center text-3xl font-venus text-white/70">
                  RG
                </div>
                <div className="flex-1">
                  <h1 className="text-2xl md:text-3xl font-venus text-white/90 mb-2">Rahul Gupta</h1>
                  <p className="text-white/40 mb-4">Computer Vision Researcher & Full Stack Engineer</p>
                  
                  {/* Technology Logos */}
                  <div className="flex flex-wrap gap-3 mb-4">
                    <div className="w-8 h-8 bg-white/10 rounded-md flex items-center justify-center" title="PyTorch">
                      <SiPytorch className="w-5 h-5 text-orange-400" />
                    </div>
                    <div className="w-8 h-8 bg-white/10 rounded-md flex items-center justify-center" title="TensorFlow">
                      <SiTensorflow className="w-5 h-5 text-orange-500" />
                    </div>
                    <div className="w-8 h-8 bg-white/10 rounded-md flex items-center justify-center" title="NVIDIA">
                      <SiNvidia className="w-5 h-5 text-green-400" />
                    </div>
                    <div className="w-8 h-8 bg-white/10 rounded-md flex items-center justify-center" title="Hugging Face">
                      <SiHuggingface className="w-5 h-5 text-yellow-300" />
                    </div>
                    <div className="w-8 h-8 bg-white/10 rounded-md flex items-center justify-center" title="Docker">
                      <SiDocker className="w-5 h-5 text-blue-400" />
                    </div>
                    <div className="w-8 h-8 bg-white/10 rounded-md flex items-center justify-center" title="OpenCV">
                      <SiOpencv className="w-5 h-5 text-red-400" />
                    </div>
                    <div className="w-8 h-8 bg-white/10 rounded-md flex items-center justify-center" title="MLOps">
                      <SiGithubactions className="w-5 h-5 text-purple-400" />
                    </div>
                  </div>
                  
                  {/* Skills Tags */}
                  <div className="flex flex-wrap gap-2">
                    <span className="px-2 py-1 bg-purple-500/20 text-purple-300 rounded-full text-xs">Computer Vision</span>
                    <span className="px-2 py-1 bg-blue-500/20 text-blue-300 rounded-full text-xs">PyTorch</span>
                    <span className="px-2 py-1 bg-green-500/20 text-green-300 rounded-full text-xs">TensorFlow</span>
                    <span className="px-2 py-1 bg-orange-500/20 text-orange-300 rounded-full text-xs">CUDA</span>
                    <span className="px-2 py-1 bg-red-500/20 text-red-300 rounded-full text-xs">Full Stack</span>
                    <span className="px-2 py-1 bg-yellow-500/20 text-yellow-300 rounded-full text-xs">Docker</span>
                    <span className="px-2 py-1 bg-indigo-500/20 text-indigo-300 rounded-full text-xs">LLMs</span>
                    <span className="px-2 py-1 bg-pink-500/20 text-pink-300 rounded-full text-xs">MLOps</span>
                  </div>
                </div>
              </div>
            </div>
            
            {/* Technical Skills Section */}
            <div className="bg-white/[0.02] backdrop-blur-sm border border-white/[0.03] rounded-lg p-6 md:p-8 shadow-[0_0_15px_rgba(255,255,255,0.02)] mb-8">
              <h2 className="text-xl font-mono text-white/40 mb-6">TECHNICAL SKILLS</h2>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <h3 className="text-white/70 font-mono text-sm mb-3">ML / AI</h3>
                  <ul className="space-y-1">
                    <li className="text-white/60 text-sm flex items-start">
                      <span className="text-white/40 mr-2">•</span>
                      <span>PyTorch & TensorFlow model development</span>
                    </li>
                    <li className="text-white/60 text-sm flex items-start">
                      <span className="text-white/40 mr-2">•</span>
                      <span>Hugging Face transformers integration</span>
                    </li>
                    <li className="text-white/60 text-sm flex items-start">
                      <span className="text-white/40 mr-2">•</span>
                      <span>Hyperparameter tuning & optimization</span>
                    </li>
                    <li className="text-white/60 text-sm flex items-start">
                      <span className="text-white/40 mr-2">•</span>
                      <span>Model quantization techniques</span>
                    </li>
                  </ul>
                </div>
                
                <div>
                  <h3 className="text-white/70 font-mono text-sm mb-3">COMPUTER VISION</h3>
                  <ul className="space-y-1">
                    <li className="text-white/60 text-sm flex items-start">
                      <span className="text-white/40 mr-2">•</span>
                      <span>OpenCV image processing pipelines</span>
                    </li>
                    <li className="text-white/60 text-sm flex items-start">
                      <span className="text-white/40 mr-2">•</span>
                      <span>Object detection & segmentation</span>
                    </li>
                    <li className="text-white/60 text-sm flex items-start">
                      <span className="text-white/40 mr-2">•</span>
                      <span>3D reconstruction algorithms</span>
                    </li>
                    <li className="text-white/60 text-sm flex items-start">
                      <span className="text-white/40 mr-2">•</span>
                      <span>Neural Radiance Fields (NeRF)</span>
                    </li>
                  </ul>
                </div>
                
                <div>
                  <h3 className="text-white/70 font-mono text-sm mb-3">GPU COMPUTING</h3>
                  <ul className="space-y-1">
                    <li className="text-white/60 text-sm flex items-start">
                      <span className="text-white/40 mr-2">•</span>
                      <span>NVIDIA CUDA infrastructure</span>
                    </li>
                    <li className="text-white/60 text-sm flex items-start">
                      <span className="text-white/40 mr-2">•</span>
                      <span>Hybrid architecture (ARM-based H200, A100)</span>
                    </li>
                    <li className="text-white/60 text-sm flex items-start">
                      <span className="text-white/40 mr-2">•</span>
                      <span>AMD-x86 optimization</span>
                    </li>
                    <li className="text-white/60 text-sm flex items-start">
                      <span className="text-white/40 mr-2">•</span>
                      <span>WebGPU acceleration</span>
                    </li>
                  </ul>
                </div>
                
                <div>
                  <h3 className="text-white/70 font-mono text-sm mb-3">MLOPS & DEPLOYMENT</h3>
                  <ul className="space-y-1">
                    <li className="text-white/60 text-sm flex items-start">
                      <span className="text-white/40 mr-2">•</span>
                      <span>Docker containerization</span>
                    </li>
                    <li className="text-white/60 text-sm flex items-start">
                      <span className="text-white/40 mr-2">•</span>
                      <span>ONNX model conversion & optimization</span>
                    </li>
                    <li className="text-white/60 text-sm flex items-start">
                      <span className="text-white/40 mr-2">•</span>
                      <span>CI/CD pipelines for ML workflows</span>
                    </li>
                    <li className="text-white/60 text-sm flex items-start">
                      <span className="text-white/40 mr-2">•</span>
                      <span>Remote GPU management</span>
                    </li>
                  </ul>
                </div>
              </div>
            </div>
            
            {/* Experience Section */}
            <div className="bg-white/[0.02] backdrop-blur-sm border border-white/[0.03] rounded-lg p-6 md:p-8 shadow-[0_0_15px_rgba(255,255,255,0.02)] mb-8">
              <h2 className="text-xl font-mono text-white/40 mb-6">EXPERIENCE</h2>
              
              {/* Experience Item */}
              <div className="mb-8">
                <div className="flex justify-between items-start mb-2">
                  <h3 className="text-lg font-venus text-white/90">Computer Vision Researcher</h3>
                  <span className="text-sm text-white/40">2022 - 2024</span>
                </div>
                <h4 className="text-white/70 mb-2">San Diego State University</h4>
                <ul className="space-y-2">
                  <li className="text-white/60 text-sm flex items-start">
                    <span className="text-white/40 mr-2">•</span>
                    <span>Developed real-time object detection systems using PyTorch and TensorFlow</span>
                  </li>
                  <li className="text-white/60 text-sm flex items-start">
                    <span className="text-white/40 mr-2">•</span>
                    <span>Implemented 3D reconstruction algorithms for spatial mapping applications</span>
                  </li>
                  <li className="text-white/60 text-sm flex items-start">
                    <span className="text-white/40 mr-2">•</span>
                    <span>Optimized computer vision algorithms for edge devices, improving inference speed by 40%</span>
                  </li>
                </ul>
              </div>
              
              {/* Walmart Labs Experience */}
              <div>
                <div className="flex justify-between items-start mb-2">
                  <h3 className="text-lg font-venus text-white/90">Full Stack Engineer</h3>
                  <span className="text-sm text-white/40">2019 - 2022</span>
                </div>
                <h4 className="text-white/70 mb-2">Walmart Labs, Sunnyvale</h4>
                <ul className="space-y-2">
                  <li className="text-white/60 text-sm flex items-start">
                    <span className="text-white/40 mr-2">•</span>
                    <span>Architected and developed end-to-end web applications with React and Node.js</span>
                  </li>
                  <li className="text-white/60 text-sm flex items-start">
                    <span className="text-white/40 mr-2">•</span>
                    <span>Implemented computer vision solutions for product recognition and inventory management</span>
                  </li>
                  <li className="text-white/60 text-sm flex items-start">
                    <span className="text-white/40 mr-2">•</span>
                    <span>Containerized applications using Docker for consistent deployment across environments</span>
                  </li>
                  <li className="text-white/60 text-sm flex items-start">
                    <span className="text-white/40 mr-2">•</span>
                    <span>Led development of GPU-accelerated image processing pipelines for content moderation</span>
                  </li>
                </ul>
              </div>
            </div>
            
            {/* Education Section */}
            <div className="bg-white/[0.02] backdrop-blur-sm border border-white/[0.03] rounded-lg p-6 md:p-8 shadow-[0_0_15px_rgba(255,255,255,0.02)] mb-8">
              <h2 className="text-xl font-mono text-white/40 mb-6">EDUCATION</h2>
              
              <div className="mb-6">
                <div className="flex justify-between items-start mb-2">
                  <h3 className="text-lg font-venus text-white/90">M.S. Computer Science</h3>
                  <span className="text-sm text-white/40">2022 - 2024</span>
                </div>
                <h4 className="text-white/70">San Diego State University</h4>
                <p className="text-white/60 text-sm mt-2">Specialization in Computer Vision and Machine Learning</p>
              </div>
              
              <div>
                <div className="flex justify-between items-start mb-2">
                  <h3 className="text-lg font-venus text-white/90">B.S. Computer Science</h3>
                  <span className="text-sm text-white/40">2014 - 2018</span>
                </div>
                <h4 className="text-white/70">San Diego State University</h4>
              </div>
            </div>
            
            {/* Projects Section */}
            <div className="bg-white/[0.02] backdrop-blur-sm border border-white/[0.03] rounded-lg p-6 md:p-8 shadow-[0_0_15px_rgba(255,255,255,0.02)]">
              <h2 className="text-xl font-mono text-white/40 mb-6">PROJECTS</h2>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="bg-white/[0.01] p-4 rounded-lg border border-white/[0.05]">
                  <h3 className="text-white/90 font-venus mb-2">Neural Radiance Fields</h3>
                  <p className="text-white/40 text-sm">Browser-based NeRF implementation using WebGPU for 3D scene reconstruction.</p>
                </div>
                
                <div className="bg-white/[0.01] p-4 rounded-lg border border-white/[0.05]">
                  <h3 className="text-white/90 font-venus mb-2">Real-time Object Tracking</h3>
                  <p className="text-white/40 text-sm">GPU-accelerated multi-object tracking system with occlusion handling.</p>
                </div>
                
                <div className="bg-white/[0.01] p-4 rounded-lg border border-white/[0.05]">
                  <h3 className="text-white/90 font-venus mb-2">WebLLM Integration</h3>
                  <p className="text-white/40 text-sm">Browser-based large language model inference using WebGPU acceleration.</p>
                </div>
                
                <div className="bg-white/[0.01] p-4 rounded-lg border border-white/[0.05]">
                  <h3 className="text-white/90 font-venus mb-2">MLOps Pipeline for CV Models</h3>
                  <p className="text-white/40 text-sm">Automated CI/CD workflow for training, testing, and deploying computer vision models.</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </main>
  );
}
