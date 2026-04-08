from models import Observation, Action
from tasks import TaskEasyBrightness

class ImageAugmentationEnv:
    def __init__(self, task=None):
        self.task = task if task else TaskEasyBrightness()
        self.reset()

    def reset(self) -> Observation:
        initial_data = self.task.get_initial_state()
        self.brightness = initial_data["avg_brightness"]
        self.noise = initial_data["noise_variance"]
        self.contrast = initial_data["contrast_ratio"]
        self.step_count = 0
        self.done = False
        self.accuracy = self._calculate_dummy_accuracy()
        return self.state()

    def state(self) -> Observation:
        return Observation(
            task_id=self.task.id,
            avg_brightness=round(self.brightness, 2),
            noise_variance=round(self.noise, 2),
            contrast_ratio=round(self.contrast, 2),
            current_accuracy=round(self.accuracy, 2),
            step_count=self.step_count
        )

    def _calculate_dummy_accuracy(self):
        # Mathematical simulation of a vision model's performance
        brightness_penalty = abs(0.5 - self.brightness) * 1.2
        noise_penalty = self.noise * 0.8
        contrast_penalty = abs(1.0 - self.contrast) * 0.5
        
        new_acc = 1.0 - brightness_penalty - noise_penalty - contrast_penalty
        return max(0.0, min(1.0, new_acc)) 

    def step(self, action: Action):
        if self.done:
            raise ValueError("Environment is done. Call reset().")

        old_accuracy = self.accuracy
        self.step_count += 1
        reward = 0.0

        # Execute Operation
        if action.operation == "increase_brightness":
            self.brightness = min(1.0, self.brightness + (action.intensity * 0.4))
        elif action.operation == "decrease_brightness":
            self.brightness = max(0.0, self.brightness - (action.intensity * 0.4))
        elif action.operation == "apply_denoise":
            self.noise = max(0.0, self.noise - (action.intensity * 0.5))
            # Heavy denoise reduces contrast slightly (real-world side effect)
            self.contrast = max(0.0, self.contrast - (action.intensity * 0.1))
        elif action.operation == "increase_contrast":
            self.contrast = min(1.0, self.contrast + (action.intensity * 0.4))
            # Increasing contrast amplifies noise slightly
            self.noise = min(1.0, self.noise + (action.intensity * 0.05))
        elif action.operation == "submit_pipeline":
            self.done = True
            
        self.accuracy = self._calculate_dummy_accuracy()

        # Dense Reward Calculation
        if not self.done:
            delta_acc = self.accuracy - old_accuracy
            reward = delta_acc * 10 if delta_acc > 0 else delta_acc * 20
        else:
            task_score = self.task.grade(self.accuracy, self.step_count)
            reward = task_score * 50 # Bonus for finishing
            
        # Timeout fail-safe
        if self.step_count >= self.task.max_steps and not self.done:
            self.done = True
            reward = -10.0

        info = {"task_score": self.task.grade(self.accuracy, self.step_count) if self.done else 0.0}
        return self.state(), round(reward, 2), self.done, info