import argparse
from runners.hf_runner import HFModelRunner
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model name or path")
    parser.add_argument("--dataset", type=str, required=True, help="Path to local dataset directory")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--anli_round", type=str, choices=["r1", "r2", "r3"], default=None,
                        help="Specify ANLI round (r1, r2, or r3). Default is sequential processing.")
    parser.add_argument("--project_dir", type=str, default="/volumes2/checkpoints", help="Directory to save checkpoints")
    args = parser.parse_args()

    dst_dir = os.path.join(args.project_dir, args.model.split("/")[-1])
    os.makedirs(dst_dir, exist_ok=True)
    print(f"Saving checkpoints to {dst_dir}")

    runner = HFModelRunner(
        model_name=args.model,
        dataset_path=args.dataset,
        output_dir=dst_dir,
        max_length=512,  # Input context length
        anli_round=args.anli_round  # Pass the ANLI round argument
    )

    # runner.train(num_train_epochs=args.epochs, per_device_batch_size=args.batch_size)
    results = runner.eval()

    print("Evaluation Results:", results)

if __name__ == "__main__":
    main()
