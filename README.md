---
title: OpenEnv Image Optimizer
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
tags:
  - openenv
---

# OpenEnv: Automated Image Augmentation Optimizer

## Description and Motivation
In modern MLOps pipelines, applying static image augmentations can destroy critical features. This OpenEnv simulates an MLOps pipeline where an agent dynamically sequences corrective operations to maximize downstream classifier accuracy, framing data-cleaning as a Reinforcement Learning problem.

## Observation and Action Spaces
**Observation Space:** A strict JSON state containing image metrics: `task_id`, `avg_brightness`, `noise_variance`, `contrast_ratio`, `step_count`, and simulated `current_accuracy`.
**Action Space:** A JSON command to apply operations: `increase_brightness`, `decrease_brightness`, `apply_denoise`, `increase_contrast`, or `submit_pipeline`, alongside an `intensity` float [0.1 - 1.0].

## Tasks & Expected Difficulty
1. **task_1_easy_brightness (Easy):** Image is heavily underexposed. The agent must increase brightness without blowing out highlights.
2. **task_2_medium_noise (Medium):** Severe static noise. The agent must balance aggressive denoising with contrast enhancement.
3. **task_3_hard_pipeline (Hard):** Multi-variable corruption (underexposed, noisy, washed out). Requires strict sequencing of multiple operations.

## Setup and Usage Instructions
**Environment Variables Required:**
To run the inference script, the following variables must be configured in your environment:
* `HF_TOKEN`: (Mandatory) Your Hugging Face / OpenAI API Key.
* `API_BASE_URL`: (Optional) Defaults to `https://api.openai.com/v1`.
* `MODEL_NAME`: (Optional) Defaults to `gpt-4o-mini`.

**Run via Docker:**
```bash
docker build -t openenv-image-optimizer .
docker run -e HF_TOKEN="your_key" openenv-image-optimizer
