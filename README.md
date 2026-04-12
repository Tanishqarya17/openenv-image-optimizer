
<img width="1812" height="343" alt="image" src="https://github.com/user-attachments/assets/f0ec218c-a86a-49d4-a044-39ccac36e1b6" />

# OpenEnv: Automated Image Augmentation Optimizer

## Description and Motivation
In modern MLOps pipelines, applying static image augmentations can destroy critical features. This OpenEnv simulates an MLOps pipeline where an agent dynamically sequences corrective operations to maximize downstream classifier accuracy, framing data-cleaning as a Reinforcement Learning problem.

## Observation and Action Spaces
**Observation Space:** A strict JSON state containing image metrics: `avg_brightness`, `noise_variance`, `contrast_ratio`, and simulated `current_accuracy`.
**Action Space:** A JSON command to apply operations: `increase_brightness`, `decrease_brightness`, `apply_denoise`, `increase_contrast`, or `submit_pipeline`, alongside an `intensity` float [0.1 - 1.0].

## Tasks & Expected Difficulty
1. **task_1_easy_brightness (Easy):** Image is heavily underexposed. The agent must increase brightness without blowing out highlights.
2. **task_2_medium_noise (Medium):** Severe static noise. The agent must balance aggressive denoising with contrast enhancement.
3. **task_3_hard_pipeline (Hard):** Multi-variable corruption (underexposed, noisy, washed out). Requires strict sequencing of multiple operations.

## Setup & Environment Variables
The inference script strictly requires the OpenAI Python Client and reads the following environment variables:
* `HF_TOKEN`: **Mandatory.** Your Hugging Face or OpenAI API key.
* `API_BASE_URL`: The API endpoint for the LLM (Defaults to `https://api.openai.com/v1`).
* `MODEL_NAME`: The model identifier (Defaults to `gpt-4o-mini`).

**Run via Docker:** `docker run -e HF_TOKEN="your_key" <image_name>`

**Run Locally:** `python inference.py` (Ensure HF_TOKEN is exported in your terminal).

## Baseline Scores
The baseline execution utilizes `gpt-4o-mini` and outputs the strictly required `[START]`, `[STEP]`, and `[END]` telemetry.

* **task_1_easy_brightness:** 0.92 / 1.0
* **task_2_medium_noise:** 0.85 / 1.0
* **task_3_hard_pipeline:** 0.81 / 1.0
