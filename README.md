# 🔍 ML Vision Project

A comprehensive computer vision showcase application with a Next.js frontend and FastAPI backend.

## 📋 Project Structure

This monorepo contains two main components:

```
/
├── ml-vision/           # Next.js frontend
│   ├── app/             # Next.js app directory
│   ├── components/      # React components
│   ├── public/          # Static assets
│   └── ...
│
└── backend-service/     # FastAPI backend
    ├── server.py        # Main server
    ├── cv_service.py    # Computer vision services
    ├── models/          # ML models
    └── ...
```

## 🚀 Development

### Prerequisites

- Node.js 18+ and pnpm for the frontend
- Python 3.9+ for the backend
- OpenAI API key for image generation

### Running Locally

1. **Start the Frontend**
   ```bash
   cd ml-vision
   pnpm install
   pnpm run dev
   ```
   The frontend will be available at `http://localhost:4000`

2. **Start the Backend**
   ```bash
   cd backend-service
   pip install -r requirements.txt
   python server.py
   ```
   The backend will be available at `http://localhost:8000`

## 🌐 Deployment

### Frontend (Vercel)

1. Connect your GitHub repository to Vercel
2. Set the root directory to `ml-vision`
3. Add environment variables:
   - `OPENAI_API_KEY`: Your OpenAI API key
   - `BACKEND_URL`: URL of your deployed backend

### Backend (Railway)

1. Connect your GitHub repository to Railway
2. Railway will automatically detect the Dockerfile in `backend-service`
3. Set the service directory to `backend-service`
4. Add environment variables if needed

## 🔧 Environment Variables

### Frontend (.env)
```
OPENAI_API_KEY=your_openai_api_key
BACKEND_URL=your_backend_url
```

### Backend (.env)
```
# Add any backend environment variables here
```

## 🧪 Features

- Image generation with DALL-E
- Edge detection
- Feature extraction
- Object detection with YOLO
- Image segmentation with SAM2
- Pose detection
- 3D visualization with Neural Radiance Fields (NeRF)

## 📚 Documentation

- [Frontend Documentation](./ml-vision/README.md)
- [Backend API Documentation](http://localhost:8000/docs) (when running locally)

---

Made with 💻 by developers who love making complex things simple.
