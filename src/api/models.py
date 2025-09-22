from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum

class BlurType(str, Enum):
    GAUSSIAN = "gaussian"
    MOTION = "motion"
    COMPRESSION = "compression"
    SCAN = "scan"
    LOWRES = "lowres"
    TEXT = "text"
    LINES = "lines"

class ModelType(str, Enum):
    UNET_BASELINE = "unet_baseline"
    DIFFUSION = "diffusion"

class DeblurRequest(BaseModel):
    image_data: str = Field(..., description="Base64 encoded image data")
    model_type: ModelType = Field(default=ModelType.DIFFUSION, description="Model to use for deblurring")
    num_inference_steps: Optional[int] = Field(default=50, description="Number of inference steps for diffusion model")
    guidance_scale: Optional[float] = Field(default=1.0, description="Guidance scale for diffusion model")

class DeblurResponse(BaseModel):
    success: bool = Field(..., description="Whether the operation was successful")
    message: str = Field(..., description="Status message")
    deblurred_image: Optional[str] = Field(None, description="Base64 encoded deblurred image")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")

class BatchDeblurRequest(BaseModel):
    image_paths: List[str] = Field(..., description="List of image file paths")
    output_dir: str = Field(..., description="Output directory for deblurred images")
    model_type: ModelType = Field(default=ModelType.DIFFUSION, description="Model to use for deblurring")
    num_inference_steps: Optional[int] = Field(default=50, description="Number of inference steps")

class BatchDeblurResponse(BaseModel):
    success: bool = Field(..., description="Whether the operation was successful")
    message: str = Field(..., description="Status message")
    processed_count: int = Field(..., description="Number of images processed")
    failed_count: int = Field(..., description="Number of images that failed")
    output_paths: List[str] = Field(..., description="List of output file paths")

class DataGenerationRequest(BaseModel):
    csv_dir: str = Field(..., description="Directory containing CSV files")
    output_dir: str = Field(..., description="Output directory for generated data")
    num_blur_variants: int = Field(default=5, description="Number of blur variants per clean image")
    blur_types: Optional[List[BlurType]] = Field(None, description="Types of blur to apply")

class DataGenerationResponse(BaseModel):
    success: bool = Field(..., description="Whether the operation was successful")
    message: str = Field(..., description="Status message")
    clean_images_generated: int = Field(..., description="Number of clean images generated")
    blur_images_generated: int = Field(..., description="Number of blur images generated")

class ModelInfo(BaseModel):
    model_type: ModelType = Field(..., description="Type of the model")
    model_path: str = Field(..., description="Path to the model file")
    is_loaded: bool = Field(..., description="Whether the model is currently loaded")
    device: str = Field(..., description="Device the model is running on")

class HealthCheckResponse(BaseModel):
    status: str = Field(..., description="Service status")
    models_available: List[ModelInfo] = Field(..., description="Available models")
    gpu_available: bool = Field(..., description="Whether GPU is available")
    memory_usage: dict = Field(..., description="Memory usage information")