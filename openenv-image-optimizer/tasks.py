class BaseTask:
    def grade(self, final_accuracy: float, step_count: int) -> float:
        if final_accuracy >= self.target_accuracy:
            # Reward efficiency: slight penalty for taking too many steps
            return max(0.0, 1.0 - (step_count * 0.02))
        return 0.0

class TaskEasyBrightness(BaseTask):
    def __init__(self):
        self.id = "task_1_easy_brightness"
        self.target_accuracy = 0.85
        self.max_steps = 5

    def get_initial_state(self):
        return {"avg_brightness": 0.1, "noise_variance": 0.0, "contrast_ratio": 1.0}

class TaskMediumNoise(BaseTask):
    def __init__(self):
        self.id = "task_2_medium_noise"
        self.target_accuracy = 0.85
        self.max_steps = 7

    def get_initial_state(self):
        return {"avg_brightness": 0.5, "noise_variance": 0.8, "contrast_ratio": 0.7}

class TaskHardPipeline(BaseTask):
    def __init__(self):
        self.id = "task_3_hard_pipeline"
        self.target_accuracy = 0.90
        self.max_steps = 10

    def get_initial_state(self):
        return {"avg_brightness": 0.2, "noise_variance": 0.6, "contrast_ratio": 0.4}

def get_all_tasks():
    return [TaskEasyBrightness(), TaskMediumNoise(), TaskHardPipeline()]