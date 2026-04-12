import math
from typing import Optional
from pydantic import BaseModel
from models import ImageAction, ImageObservation

class ActionResult(BaseModel):
    observation: ImageObservation
    reward: float
    done: bool
    error: Optional[str] = None

class ImageOptimizerEnv:
    def __init__(self, task_id: str):
        self.task_id = task_id
        self.max_steps = 8
        self.reset_state()

    def reset_state(self):
        self.step_count = 0
        self.done = False
        
        # Initialize tasks with distinct mathematical corruption profiles
        if self.task_id == "task_1_easy_brightness":
            self.brightness = 0.2
            self.noise = 0.05
            self.contrast = 0.8
        elif self.task_id == "task_2_medium_noise":
            self.brightness = 0.6
            self.noise = 0.85
            self.contrast = 0.4
        elif self.task_id == "task_3_hard_pipeline":
            self.brightness = 0.1
            self.noise = 0.9
            self.contrast = 0.2
        else:
            self.brightness = 0.5
            self.noise = 0.5
            self.contrast = 0.5

        self.accuracy = self._calculate_accuracy()
        return self._get_obs()

    async def reset(self) -> ActionResult:
        obs = self.reset_state()
        return ActionResult(observation=obs, reward=0.0, done=False)

    def _calculate_accuracy(self) -> float:
            # Ideal states: Brightness ~0.6, Noise ~0.0, Contrast ~0.9
            b_penalty = abs(0.6 - self.brightness) * 1.5
            n_penalty = self.noise * 2.0
            c_penalty = abs(0.9 - self.contrast) * 1.2
            
            base = 1.0 - (b_penalty + n_penalty + c_penalty)
            
            # Strict OpenEnv Clamping: Must be between 0 and 1 (exclusive)
            return max(0.01, min(0.99, base))

    def _get_obs(self) -> ImageObservation:
        return ImageObservation(
            task_id=self.task_id,
            avg_brightness=round(self.brightness, 2),
            noise_variance=round(self.noise, 2),
            contrast_ratio=round(self.contrast, 2),
            current_accuracy=round(self.accuracy, 3),
            step_count=self.step_count
        )

    async def step(self, action: ImageAction) -> ActionResult:
        if self.done:
            return ActionResult(observation=self._get_obs(), reward=0.0, done=True, error="Episode already finished.")

        self.step_count += 1
        prev_acc = self.accuracy
        op = action.operation
        i = action.intensity

        # Mathematical simulation of cv2 operations
        if op == "increase_brightness":
            self.brightness = min(1.0, self.brightness + (0.4 * i))
        elif op == "decrease_brightness":
            self.brightness = max(0.0, self.brightness - (0.4 * i))
        elif op == "apply_denoise":
            self.noise = max(0.0, self.noise - (0.5 * i))
            self.contrast = max(0.0, self.contrast - (0.1 * i)) # Denoising inherently blurs
        elif op == "increase_contrast":
            self.contrast = min(1.0, self.contrast + (0.4 * i))
            self.noise = min(1.0, self.noise + (0.1 * i)) # Boosting contrast highlights noise
        
        self.accuracy = self._calculate_accuracy()
        
        if op == "submit_pipeline" or self.step_count >= self.max_steps:
            self.done = True
            # Final grading reward
            final_reward = self.accuracy * 10.0
            return ActionResult(observation=self._get_obs(), reward=final_reward, done=True)

        # Dense partial reward: positive for acc gain, negative for acc loss
        delta = self.accuracy - prev_acc
        reward = delta * 5.0 
        
        return ActionResult(observation=self._get_obs(), reward=reward, done=False)

    async def close(self):
        pass