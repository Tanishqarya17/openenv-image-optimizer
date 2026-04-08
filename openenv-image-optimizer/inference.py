import os
import json
from openai import OpenAI
from environment import ImageAugmentationEnv
from tasks import get_all_tasks
from models import Action


API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    HF_TOKEN = os.environ.get("OPENAI_API_KEY") 
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN environment variable is missing and mandatory.")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

def run_baseline():
    tasks = get_all_tasks()
    benchmark_name = "image-augmentation-optimizer"

    for task in tasks:
        env = ImageAugmentationEnv(task=task)
        obs = env.reset()
        done = False
        

        print(f"[START] task={task.id} env={benchmark_name} model={MODEL_NAME}")
        
        system_prompt = """You are an MLOps Agent optimizing an image pipeline. 
        Analyze the Observation state. Your goal is to get 'current_accuracy' > 0.85.
        Ideal metrics: avg_brightness ~0.5, noise_variance ~0.0, contrast_ratio ~1.0.
        Output ONLY valid JSON matching the Action schema. Do not explain your reasoning."""

        episode_rewards = []
        error_msg = "null"

        while not done:
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Observation: {json.dumps(obs.model_dump())}\nAction Schema: {Action.model_json_schema()}"}
                    ],
                    response_format={ "type": "json_object" }
                )
                action_data = json.loads(response.choices[0].message.content)
                action = Action(**action_data)
                

                action_str = f"{action.operation}({action.intensity})"
                
                obs, reward, done, info = env.step(action)
                
                episode_rewards.append(f"{reward:.2f}")
                done_str = "true" if done else "false"
                

                print(f"[STEP] step={env.step_count} action={action_str} reward={reward:.2f} done={done_str} error=null")
                
            except Exception as e:

                error_msg = str(e).replace('\n', ' ')
                done = True
                done_str = "true"
                print(f"[STEP] step={env.step_count + 1} action=error reward=0.00 done={done_str} error={error_msg}")
                break


        final_score = env.task.grade(env.accuracy, env.step_count)
        success_str = "true" if final_score > 0.0 else "false"
        rewards_str = ",".join(episode_rewards) if episode_rewards else "0.00"
        

        print(f"[END] success={success_str} steps={len(episode_rewards)} rewards={rewards_str}")

if __name__ == "__main__":
    run_baseline()