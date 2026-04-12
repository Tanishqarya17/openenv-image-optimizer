import asyncio
import os
import textwrap
from typing import List, Optional
import json
from openai import OpenAI
from environment import ImageOptimizerEnv
from models import ImageAction

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
API_KEY = os.getenv("HF_TOKEN")

if not API_KEY:
    raise ValueError("HF_TOKEN environment variable is required")

BENCHMARK = "openenv-image-optimizer"
MAX_STEPS = 8

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

def get_model_action(client: OpenAI, obs) -> ImageAction:
    system_prompt = textwrap.dedent("""
        You are an MLOps orchestrator fixing corrupted image datasets.
        Review the current image metrics.
        Target metrics: Brightness ~0.6, Noise ~0.0, Contrast ~0.9.
        Available operations: increase_brightness, decrease_brightness, apply_denoise, increase_contrast, submit_pipeline.
        You must output ONLY valid JSON matching this schema: {"operation": "string", "intensity": 0.5}
        When accuracy is high enough (>0.85), use 'submit_pipeline'.
    """).strip()

    user_prompt = f"Current State: {obs.model_dump_json()}"

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={ "type": "json_object" }
        )
        response_text = completion.choices[0].message.content
        action_dict = json.loads(response_text)
        return ImageAction(**action_dict)
    except Exception as e:
        # Fallback to prevent crash
        return ImageAction(operation="submit_pipeline", intensity=1.0)

async def run_task(task_id: str, client: OpenAI):
    env = ImageOptimizerEnv(task_id=task_id)
    history, rewards = [], []
    steps_taken, score = 0, 0.0
    
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset()
        obs = result.observation

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            action = get_model_action(client, obs)
            action_str = f"{action.operation}({action.intensity})"
            
            result = await env.step(action)
            obs = result.observation
            reward = result.reward or 0.0
            done = result.done

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action_str, reward=reward, done=done, error=result.error)

            if done:
                # The final normalized score is the accuracy, strictly clamped
                score = max(0.01, min(0.99, obs.current_accuracy)) 
                break

        success = score >= 0.85

    finally:
        await env.close()
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    tasks = ["task_1_easy_brightness", "task_2_medium_noise", "task_3_hard_pipeline"]
    for task in tasks:
        await run_task(task, client)

if __name__ == "__main__":
    asyncio.run(main())