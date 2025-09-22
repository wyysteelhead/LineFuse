from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch
import base64
import io
from PIL import Image
import numpy as np
from pathlib import Path
import logging

from .models import (
    DeblurRequest, DeblurResponse, BatchDeblurRequest, BatchDeblurResponse,
    DataGenerationRequest, DataGenerationResponse, HealthCheckResponse,
    ModelType
)
from .service import DeblurService
from ..data.clean_chart_generator import CleanChartGenerator
from ..data.blur_generator import BlurGenerator
from ..data.dataset_builder import DatasetBuilder

app = FastAPI(
    title="LineFuse API",
    description="Diffusion-based Image Deblurring for Spectral Line Charts",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

deblur_service = DeblurService()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.on_event("startup")
async def startup_event():
    logger.info("Starting LineFuse API server...")
    
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down LineFuse API server...")

@app.get("/", response_model=dict)
async def root():
    return {
        "message": "LineFuse API - Diffusion-based Image Deblurring",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    try:
        models_info = deblur_service.get_model_info()
        memory_usage = deblur_service.get_memory_usage()
        
        return HealthCheckResponse(
            status="healthy",
            models_available=models_info,
            gpu_available=torch.cuda.is_available(),
            memory_usage=memory_usage
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.post("/models/load")
async def load_model(model_type: ModelType, model_path: str):
    try:
        success = deblur_service.load_model(model_type, model_path)
        if success:
            return {"message": f"Model {model_type.value} loaded successfully"}
        else:
            raise HTTPException(status_code=400, detail=f"Failed to load model {model_type.value}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/models/unload")
async def unload_model(model_type: ModelType):
    try:
        success = deblur_service.unload_model(model_type)
        if success:
            return {"message": f"Model {model_type.value} unloaded successfully"}
        else:
            raise HTTPException(status_code=400, detail=f"Model {model_type.value} was not loaded")
    except Exception as e:
        logger.error(f"Failed to unload model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/deblur", response_model=DeblurResponse)
async def deblur_image(request: DeblurRequest):
    try:
        response = deblur_service.process_single_image(
            image_data=request.image_data,
            model_type=request.model_type,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale
        )
        return response
    except Exception as e:
        logger.error(f"Deblur request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/deblur/upload")
async def deblur_upload(
    file: UploadFile = File(...),
    model_type: ModelType = ModelType.DIFFUSION,
    num_inference_steps: int = 50,
    guidance_scale: float = 1.0
):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image_array = np.array(image)
        
        image_base64 = base64.b64encode(contents).decode()
        
        response = deblur_service.process_single_image(
            image_data=image_base64,
            model_type=model_type,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        )
        
        return response
    except Exception as e:
        logger.error(f"Upload deblur failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/deblur/batch", response_model=BatchDeblurResponse)
async def deblur_batch(request: BatchDeblurRequest):
    try:
        response = deblur_service.process_batch_images(
            image_paths=request.image_paths,
            output_dir=request.output_dir,
            model_type=request.model_type,
            num_inference_steps=request.num_inference_steps
        )
        return response
    except Exception as e:
        logger.error(f"Batch deblur failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/data/generate", response_model=DataGenerationResponse)
async def generate_data(request: DataGenerationRequest):
    try:
        chart_generator = CleanChartGenerator()
        blur_generator = BlurGenerator()
        
        clean_dir = Path(request.output_dir) / "clean"
        blur_dir = Path(request.output_dir) / "blur"
        
        clean_dir.mkdir(parents=True, exist_ok=True)
        blur_dir.mkdir(parents=True, exist_ok=True)
        
        chart_generator.batch_process(request.csv_dir, clean_dir)
        
        blur_generator.batch_generate_blur(
            clean_dir, 
            blur_dir, 
            num_variants_per_image=request.num_blur_variants
        )
        
        clean_count = len(list(clean_dir.glob("*.png")))
        blur_count = len(list(blur_dir.glob("*.png")))
        
        return DataGenerationResponse(
            success=True,
            message=f"Generated {clean_count} clean and {blur_count} blur images",
            clean_images_generated=clean_count,
            blur_images_generated=blur_count
        )
        
    except Exception as e:
        logger.error(f"Data generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/data/build-dataset")
async def build_dataset(
    clean_dir: str,
    blur_dir: str,
    output_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1
):
    try:
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")
        
        dataset_builder = DatasetBuilder()
        dataset_builder.split_data(
            clean_dir=clean_dir,
            blur_dir=blur_dir,
            output_dir=output_dir,
            split_ratios=(train_ratio, val_ratio, test_ratio)
        )
        
        stats = dataset_builder.validate_dataset(output_dir)
        
        return {
            "message": "Dataset built successfully",
            "statistics": stats
        }
        
    except Exception as e:
        logger.error(f"Dataset building failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)