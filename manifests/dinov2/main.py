# Import required libraries
from contextlib import asynccontextmanager
import io
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
import uvicorn
from PIL import Image

# Global variable for the model
model = None

def get_model():
    global model
    if model is None:
        print("Loading DINOv2 model from torch hub...")
        # Load the DINOv2 model from PyTorch Hub
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
            print("Model moved to CUDA.")
        else:
            print("Model running on CPU.")
    return model

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-load the model at startup to leverage persistent cache
    get_model()
    yield
    # Clean up resources during shutdown if needed
    print("Shutting down application...")

app = FastAPI(title="DINOv2 API", lifespan=lifespan)

@app.get("/health")
def health_check():
    # Return the API status
    return {"status": "healthy", "model": "dinov2_vits14", "cuda": torch.cuda.is_available()}

@app.post("/extract")
async def extract_features(file: UploadFile = File(...)):
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")
    
    try:
        # Read image content
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Get model
        net = get_model()
        
        # Use standard PyTorch transforms required by DINOv2
        import torchvision.transforms as T
        transform = T.Compose([
            T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        tensor = transform(image).unsqueeze(0) # Add batch dimension
        
        if torch.cuda.is_available():
            tensor = tensor.cuda()
            
        # Extract features (inference)
        with torch.no_grad():
            features = net(tensor)
            
        # Convert tensor to a flat list of floats
        feature_vector = features.squeeze().tolist()
        
        return {
            "filename": file.filename,
            "feature_dim": len(feature_vector),
            "features": feature_vector[:10] # Return only the first 10 values for brevity
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Start the server on port 8080
    uvicorn.run(app, host="0.0.0.0", port=8080)
