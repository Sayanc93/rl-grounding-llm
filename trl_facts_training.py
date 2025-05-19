import os
import logging
import argparse
import re
import torch  # type: ignore
import wandb
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore
from trl_facts_utils import extract_snippet, advanced_verify_claim, parse_claims_and_evidence

# Try importing GRPOTrainer from TRL; fallback to PPOTrainer if not available
try:
    from trl import GRPOTrainer, GRPOConfig  # type: ignore
except ImportError:
    raise ImportError("GRPOTrainer not found. Please install TRL library.")

# Try to enable Liger kernels if available
try:
    import liger  # type: ignore
    liger.enable_liger_kernels()
    logging.info('Liger kernels enabled for memory efficiency.')
except ImportError:
    logging.info('Liger not available; proceeding without Liger kernels.')

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

def load_data(data_dir):
    train_set = load_dataset('json', data_files=f"{data_dir}/train.jsonl")['train']
    train_set = train_set.map(lambda x: { # type: ignore
        'prompt': [
            {'role': 'user', 'content': x['full_prompt']}
        ],
    })
    return train_set

def compute_format_reward(output):
    """
    Check if the output adheres to the required augmented format with <claim> and <evidence_pointer> tags.
    Returns +1 if at least one valid pair is found, 0 otherwise.
    """
    claims = re.findall(r"<claim>(.*?)</claim>", output, re.DOTALL)
    evidence = re.findall(r"<evidence_pointer>(.*?)</evidence_pointer>", output, re.DOTALL)
    if claims and evidence and len(claims) == len(evidence):
        return 1.0
    return 0.0


# Task reward function
def compute_task_reward(prediction, ground_truth):
    """
    Compute a simple task reward based on overlap between the prediction and ground truth.
    """
    words = ground_truth.split()
    if not words:
        return 0.0
    match_count = sum(1 for word in words if word in prediction)
    return match_count / len(words)

# Format reward function
def format_reward_func(prompts, completions, **kwargs):
    """
    Reward function that checks if the output adheres to the required format
    with <claim> and <evidence_pointer> tags.
    """
    return [compute_format_reward(completion[0]['content']) * 0.5 for completion in completions]

# Grounding reward function
class GroundingRewardFunc:
    def __init__(self, verifier_tokenizer, verifier_model):
        self.verifier_tokenizer = verifier_tokenizer
        self.verifier_model = verifier_model

    def __call__(self, prompts, completions, **kwargs):
        rewards = []
        for i, completion in enumerate(completions):
            triplets = parse_claims_and_evidence(completion[0]['content'])
            r_grounding = 0.0
            for claim, evidence_pointer, _ in triplets:
                context_document = prompts[i][0]['content']
                snippet = extract_snippet(context_document, evidence_pointer)
                r_grounding += advanced_verify_claim(claim, snippet, self.verifier_tokenizer, self.verifier_model)
            if triplets:
                r_grounding /= len(triplets)
            rewards.append(r_grounding)
        return rewards
    
    @property
    def __name__(self):
        return "grounding_reward_func"

# Task reward function
def task_reward_func(prompts, completions, **kwargs):
    """
    Reward function that computes overlap between prediction and ground truth.
    """
    return [compute_task_reward(completion[0]['content'], ground_truth[0]['content']) 
            for completion, ground_truth in zip(completions, prompts)]

def main():
    parser = argparse.ArgumentParser(description="Run VeGoR training on the FACTS dataset using GRPO and Qwen models.")
    parser.add_argument("--policy_model_name", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="Huggingface model ID for the Qwen policy model.")
    parser.add_argument("--verifier_model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="Huggingface model ID for the Qwen verifier model.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for RL training.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--w_grounding", type=float, default=1.0, help="Weight for grounding reward.")
    parser.add_argument("--w_task", type=float, default=1.0, help="Weight for task reward.")
    parser.add_argument("--w_format", type=float, default=0.5, help="Weight for format reward.")
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory for results.")
    parser.add_argument("--run_name", type=str, default="trl-facts-grpo-training", help="Run name for wandb.")
    args = parser.parse_args()

    # Initialize wandb
    wandb.init(project="facts-grpo-training", config={
        "policy_model_name": args.policy_model_name,
        "verifier_model_name": args.verifier_model_name,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "w_grounding": args.w_grounding,
        "w_task": args.w_task,
        "w_format": args.w_format
    })

    # Load dataset from JSONL files
    train_set = load_data(DATA_DIR)

    # Load tokenizers and models for policy and verifier
    policy_tokenizer = AutoTokenizer.from_pretrained(args.policy_model_name)
    policy_model = AutoModelForCausalLM.from_pretrained(args.policy_model_name)

    verifier_tokenizer = AutoTokenizer.from_pretrained(args.verifier_model_name)
    verifier_model = AutoModelForCausalLM.from_pretrained(args.verifier_model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_model.to(device)
    verifier_model.to(device)

    # Setup GRPO trainer from TRL (see https://huggingface.co/docs/trl for details)
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        run_name=args.run_name,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        loss_type="dr_grpo",
        scale_rewards=False,
        bf16=True,
        gradient_accumulation_steps=4,
        num_generations=8,
        logging_steps=1,
        max_grad_norm=0.1,
        max_prompt_length=131072,
        max_completion_length=8192,
        weight_decay = 0.1,
        warmup_ratio = 0.1,
        use_liger_kernel=True,
        report_to="wandb",
        data_seed=42,
        torch_compile=True
    )

    # Create reward functions list with weights
    grounding_reward_func = GroundingRewardFunc(verifier_tokenizer, verifier_model)
    reward_funcs = [
        format_reward_func,
        grounding_reward_func,
        task_reward_func
    ]
    
    trainer = GRPOTrainer(
        model=policy_model,
        args=training_args,
        reward_funcs=reward_funcs,
        train_dataset=train_set
    )

    trainer.train()
    
    # Save model
    policy_model.save_pretrained("trained_qwen_policy")
    logging.info("Training complete and models saved.")
    wandb.finish()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main() 