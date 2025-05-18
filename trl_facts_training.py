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
    data_files = {
        "train": f"{data_dir}/train.jsonl",
        "eval": f"{data_dir}/eval.jsonl",
        "test": f"{data_dir}/test.jsonl"
    }
    return load_dataset('json', data_files=data_files)

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


def compute_task_reward(prediction, ground_truth):
    """
    Compute a simple task reward based on overlap between the prediction and ground truth.
    """
    words = ground_truth.split()
    if not words:
        return 0.0
    match_count = sum(1 for word in words if word in prediction)
    return match_count / len(words)

def main():
    parser = argparse.ArgumentParser(description="Run VeGoR training on the FACTS dataset using GRPO and Qwen models.")
    parser.add_argument("--policy_model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Huggingface model ID for the Qwen policy model.")
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
    train_set, eval_set, test_set = load_data(DATA_DIR)

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
        loss_type="dr_grpo",
        use_vllm=True,
        scale_rewards=False,
        bf16=True,
        gradient_accumulation_steps=4,
        num_generations=8,
        logging_steps=1,
        max_grad_norm=0.1,
        max_new_tokens=8192,
        weight_decay = 0.1,
        warmup_ratio = 0.1,
        use_liger_kernel=True,
        report_to="wandb",
        data_seed=42,
        torch_compile=True,
        vllm_mode="colocate",
    )
    trainer = GRPOTrainer(
        model=policy_model,
        tokenizer=policy_tokenizer,
        args=training_args,
        optimizers=["fused_adamw", "linear"],
        optimize_cuda_cache=True,
    )

    # Training loop
    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch+1}/{args.epochs}")
        wandb.log({"epoch": epoch+1})
        epoch_rewards = []
        for batch_start in range(0, len(train_set), args.batch_size):
            batch = train_set[batch_start:batch_start+args.batch_size]
            prompts = []
            ground_truths = []
            documents = []
            for sample in batch:
                # Each sample is assumed to have 'document', 'query', and 'ground_truth' fields
                prompt = (
                    f"Document: {sample['document']}\n"
                    f"User Query: {sample['query']}\n"
                    "Answer in format: <claim>...</claim> <evidence_pointer>...</evidence_pointer> <confidence>...</confidence> ..."
                )
                prompts.append(prompt)
                ground_truths.append(sample['ground_truth'])
                documents.append(sample['document'])
            
            # Generate outputs from the policy model
            outputs = trainer.generate(prompts)
            rewards_list = []
            for i, output in enumerate(outputs):
                decoded_output = policy_tokenizer.decode(output, skip_special_tokens=True)
                r_format = compute_format_reward(decoded_output)
                triplets = parse_claims_and_evidence(decoded_output)
                r_grounding = 0.0
                for claim, evidence_pointer, _ in triplets:
                    snippet = extract_snippet(documents[i], evidence_pointer)
                    r_grounding += advanced_verify_claim(claim, snippet, verifier_tokenizer, verifier_model)
                if triplets:
                    r_grounding /= len(triplets)
                r_task = compute_task_reward(decoded_output, ground_truths[i])
                total_reward = args.w_grounding * r_grounding + args.w_task * r_task + args.w_format * r_format
                rewards_list.append(total_reward)
                epoch_rewards.append(total_reward)
            
            # Update the policy using GRPO step with computed rewards
            trainer.step(prompts, outputs, rewards_list)
            batch_avg_reward = sum(rewards_list)/len(rewards_list) if rewards_list else 0.0
            logging.info(f"Processed batch {batch_start // args.batch_size + 1}, average reward: {batch_avg_reward:.4f}")
            wandb.log({"batch": batch_start // args.batch_size + 1, "batch_avg_reward": batch_avg_reward})
        
        epoch_avg_reward = sum(epoch_rewards)/len(epoch_rewards) if epoch_rewards else 0.0
        logging.info(f"Epoch {epoch+1} average reward: {epoch_avg_reward:.4f}")
        wandb.log({"epoch": epoch+1, "epoch_avg_reward": epoch_avg_reward})
        # Optionally, evaluation on the eval_set can be performed per epoch
    
    # Final evaluation on test set
    logging.info("Starting final evaluation on test set")
    test_prompts = []
    test_ground_truths = []
    for sample in test_set:
        prompt = (
            f"Document: {sample['document']}\n"
            f"User Query: {sample['query']}\n"
            "Answer in format: <claim>...</claim> <evidence_pointer>...</evidence_pointer> <confidence>...</confidence> ..."
        )
        test_prompts.append(prompt)
        test_ground_truths.append(sample['ground_truth'])
    test_outputs = trainer.generate(test_prompts, max_new_tokens=8192)
    test_rewards = []
    for i, output in enumerate(test_outputs):
        decoded_output = policy_tokenizer.decode(output, skip_special_tokens=True)
        r_format = compute_format_reward(decoded_output)
        triplets = parse_claims_and_evidence(decoded_output)
        r_grounding = 0.0
        for claim, evidence_pointer, _ in triplets:
            snippet = extract_snippet(test_set[i]['document'], evidence_pointer)
            r_grounding += advanced_verify_claim(claim, snippet, verifier_tokenizer, verifier_model)
        if triplets:
            r_grounding /= len(triplets)
        r_task = compute_task_reward(decoded_output, test_ground_truths[i])
        total_reward = args.w_grounding * r_grounding + args.w_task * r_task + args.w_format * r_format
        test_rewards.append(total_reward)
    avg_test_reward = sum(test_rewards) / len(test_rewards) if test_rewards else 0.0
    logging.info(f"Test set average reward: {avg_test_reward:.4f}")
    wandb.log({"avg_test_reward": avg_test_reward})
    
    # Save model
    policy_model.save_pretrained("trained_qwen_policy")
    logging.info("Training complete and models saved.")
    wandb.finish()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main() 