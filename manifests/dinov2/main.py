from contextlib import asynccontextmanager
import io
import os
from PIL import Image
import torch
import torchvision.transforms as T
from fastapi import FastAPI, File, HTTPException, UploadFile
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
import uvicorn

# Configure OpenTelemetry tracing
service_name = os.getenv("OTEL_SERVICE_NAME", "dinov2-cpu")
otlp_endpoint = os.getenv(
    "OTEL_EXPORTER_OTLP_ENDPOINT",
    "http://jaeger.tracing.svc.cluster.local:4317",
)

resource = Resource.create({"service.name": service_name})
tracer_provider = TracerProvider(resource=resource)
otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
tracer_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
trace.set_tracer_provider(tracer_provider)

tracer = trace.get_tracer(__name__)

# Global variable for the model
model = None

# Standard PyTorch transforms required by DINOv2
transform = T.Compose([
    T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def get_model():
    global model
    if model is None:
        print("Loading DINOv2 model from torch hub...")
        # Load the DINOv2 model from PyTorch Hub
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
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
    # Flush remaining spans and clean up resources during shutdown
    tracer_provider.shutdown()
    print("Shutting down application...")


app = FastAPI(title="DINOv2 API", lifespan=lifespan)
FastAPIInstrumentor.instrument_app(app)


@app.get("/health")
def health_check():
    # Return the API status
    return {
        "status": "healthy",
        "model": "dinov2_vits14",
        "cuda": torch.cuda.is_available(),
    }


@app.post("/extract")
async def extract_features(file: UploadFile = File(...)):
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    try:
        with tracer.start_as_current_span("preprocess_image") as span:
            span.set_attribute("file.name", file.filename or "unknown")
            span.set_attribute("file.content_type", file.content_type)
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            tensor = transform(image).unsqueeze(0)
            if torch.cuda.is_available():
                tensor = tensor.cuda()

        # Get model
        net = get_model()

        # Extract features (inference)
        with tracer.start_as_current_span("dinov2_inference") as span:
            span.set_attribute("model.name", "dinov2_vits14")
            span.set_attribute("device", "cuda" if torch.cuda.is_available() else "cpu")
            with torch.no_grad():
                features = net(tensor)

        # Convert tensor to a flat list of floats
        feature_vector = features.squeeze().tolist()

        return {
            "filename": file.filename,
            "feature_dim": len(feature_vector),
            "features": feature_vector[:10],
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # Start the server on port 8080
    uvicorn.run(app, host="0.0.0.0", port=8080)
