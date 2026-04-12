from pydantic import BaseModel, Field
from typing import Literal

class ImageAction(BaseModel):
    operation: Literal[
        "increase_brightness", 
        "decrease_brightness", 
        "apply_denoise", 
        "increase_contrast", 
        "submit_pipeline"
    ]
    intensity: float = Field(ge=0.1, le=1.0, description="Strength of the operation (0.1 to 1.0)")

class ImageObservation(BaseModel):
    task_id: str
    avg_brightness: float
    noise_variance: float
    contrast_ratio: float
    current_accuracy: float
    step_count: int