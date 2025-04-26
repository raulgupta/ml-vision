# ML Vision - Computer Vision Showcase 🎮

## What We Have 🎯

A comprehensive computer vision showcase application that demonstrates various CV techniques:
- Edge detection
- Feature extraction
- Object detection
- Image segmentation
- Pose detection
- 3D visualization with Neural Radiance Fields (NeRF)

## How It Works ⚙️

### The Computer Vision Pipeline 🖥️
- Upload or generate images using OpenAI
- Apply various computer vision algorithms
- Visualize the results with interactive controls
- Explore 3D models using WebGPU acceleration

### The Key Features 🧩

#### Image Generation 🖼️
- Generate images using OpenAI's DALL-E
- Add negative prompts to refine results
- Optimized for computer vision processing

#### Computer Vision Tools 🔍
- Edge detection with adjustable parameters
- Feature extraction using SIFT or ORB
- Object detection with YOLO
- Image segmentation with SAM2
- Pose detection for human figures

#### 3D Visualization 🌐
- Neural Radiance Fields (NeRF) rendering
- WebGPU acceleration when available
- Interactive parameter controls

## Project Structure 📁

```
ml-vision/
│
├── components/           # React components
│   ├── CVService.ts        # Computer vision service
│   ├── VolumeRenderer.tsx  # 3D NeRF renderer
│   ├── PersistentInput.tsx # Image prompt input
│   ├── ImageGenTitle.tsx   # Animated title
│   └── Navbar.tsx          # Navigation bar
│
├── app/                 # Next.js app
│   ├── page.tsx         # Home page with image generation
│   ├── cv/              # Computer vision page
│   ├── showcase/        # NeRF showcase page
│   ├── resume/          # Resume page
│   ├── api/             # API routes
│   │   ├── cv/          # Computer vision API routes
│   │   └── generate-image/ # Image generation API
│   └── layout.tsx       # Main layout
│
├── public/              # Static files
│   ├── truck_rgba_slices/ # NeRF volume slices
│   └── models/          # 3D models
│
└── backend-service/     # Python FastAPI backend
    ├── server.py        # Main server
    ├── cv_service.py    # CV processing
    └── models/          # ML models
```

## Quick Start Guide 🚀

1. **Start the Frontend**
   ```bash
   cd ml-vision
   pnpm run dev
   ```
   This starts your frontend at `http://localhost:4000`

2. **Start the Backend**
   ```bash
   cd backend-service
   ./start.sh
   ```
   This starts the FastAPI backend at `http://localhost:8000`

3. **Use the Application**
   - Generate images with DALL-E
   - Apply computer vision algorithms
   - Explore the 3D NeRF visualization
   - Check out the resume page

## Example Features 💭

Try these features:
- Generate an image of a cityscape
- Apply edge detection with different thresholds
- Detect objects in your generated images
- Segment images using SAM2
- Explore the 3D truck model in the showcase

## Design Choices 🎨

We use:
- Glass-like effects for a modern look
- Military-inspired dark theme with mesh overlays
- Interactive controls for algorithm parameters
- Responsive design for all screen sizes
- WebGPU acceleration for 3D rendering

## Backend Architecture 📋

The application uses a hybrid architecture:
- Next.js frontend with API routes
- FastAPI Python backend for computer vision processing
- OpenAI integration for image generation
- PyTorch and OpenCV for computer vision algorithms
- WebGPU for 3D rendering acceleration

## Future Ideas 🔮

We plan to add:
- More computer vision algorithms
- Additional 3D models for NeRF visualization
- Real-time video processing
- Mobile device camera integration
- Collaborative annotation tools

Explore the power of computer vision with ML Vision! 🔍

---

Made with 💻 by developers who love making complex things simple.
