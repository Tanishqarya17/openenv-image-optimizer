# OpenEnv: Automated Image Augmentation Policy Optimizer

## Description and Motivation
When deploying Computer Vision models in real-world scenarios (e.g., edge IoT cameras, medical imaging), input data is frequently corrupted by poor lighting, sensor noise, or weather conditions. Standard, hardcoded image processing pipelines fail to adapt to dynamic corruption. 

This OpenEnv simulates an MLOps pipeline where an agent must dynamically analyze image metrics and apply a sequence of corrective operations (brightness adjustments, denoising, contrast enhancements) to maximize the accuracy of a downstream classifier. It frames data-cleaning as a Reinforcement Learning problem, where the agent is penalized for destructive operations and rewarded for recovering classifier accuracy.

## Observation and Action Spaces

### Observation Space
The agent receives a strict JSON state containing extracted metrics of the current image batch, rather than raw pixel tensors.
* `task_id` (str): The current difficulty level being evaluated.
* `avg_brightness` (float): Current image brightness [0.0 (black) to 1.0 (white)].
* `noise_variance` (float): Current noise level [0.0 (clean) to 1.0 (heavy static)].
* `contrast_ratio` (float): Current contrast [0.0 (washed out) to 1.0 (sharp)].
* `current_accuracy` (float): Simulated accuracy of the downstream classifier [0.0 to 1.0]. Target is > 0.85.
* `step_count` (int): Number of processing operations applied so far.

### Action Space
The agent outputs a strict JSON command to apply OpenCV-style transformations.
* `operation` (Literal): Must be one of `increase_brightness`, `decrease_brightness`, `apply_denoise`, `increase_contrast`, or `submit_pipeline`.
* `intensity` (float): The strength of the operation, ranging from `0.1` (slight adjustment) to `1.0` (maximum application).

## Tasks & Expected Difficulty

1. **Task 1: The Night Vision Problem (`task_1_easy_brightness`)**
   * **Difficulty:** Easy
   * **Description:** The initial image state is heavily underexposed (brightness = 0.1) but otherwise clean. The agent must successfully increase brightness without blowing out the highlights to achieve >85% accuracy.

2. **Task 2: The Noisy Sensor (`task_2_medium_noise`)**
   * **Difficulty:** Medium
   * **Description:** The image has standard exposure but severe static noise (noise = 0.8) and degraded contrast. The agent must balance aggressive denoising, which inherently blurs the image and reduces contrast, with contrast enhancement.

3. **Task 3: The Mixed Corruption Pipeline (`task_3_hard_pipeline`)**
   * **Difficulty:** Hard
   * **Description:** The image suffers from multi-variable corruption: underexposed, noisy, and washed out. The agent must sequence multiple operations in the correct order to reach >90% accuracy before the step limit expires.

## Setup and Usage Instructions

### Local Development
1. Clone the repository and navigate to the root directory.
2. Create and activate a Python 3.10 virtual environment.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
