from pydantic import BaseModel, Field
from typing import Literal

class Observation(BaseModel):
    task_id: str = Field(..., description="The current task identifier.")
    avg_brightness: float = Field(..., description="Current image brightness. 0.0 is black, 1.0 is white. Ideal is ~0.5.")
    noise_variance: float = Field(..., description="Current noise level. 0.0 is clean, 1.0 is heavy static.")
    contrast_ratio: float = Field(..., description="Current contrast. 0.0 is washed out, 1.0 is sharp. Ideal is ~1.0.")
    current_accuracy: float = Field(..., description="Accuracy of the dummy classifier (0.0 to 1.0). Target is > 0.85.")
    step_count: int = Field(..., description="Number of augmentations applied so far.")

class Action(BaseModel):
    operation: Literal[
        "increase_brightness", 
        "decrease_brightness", 
        "apply_denoise", 
        "increase_contrast",
        "submit_pipeline"
    ] = Field(..., description="The image processing operation to apply.")
    intensity: float = Field(
        ..., ge=0.1, le=1.0, description="How strongly to apply the operation. 0.1 is slight, 1.0 is maximum."
    )