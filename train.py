import argparse
from runners.hf_runner import HFModelRunner

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model name or path")
    parser.add_argument("--dataset", type=str, required=True, help="Path to local dataset directory")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    runner = HFModelRunner(
        model_name=args.model,
        dataset_path=args.dataset,
    )

    runner.train(num_train_epochs=args.epochs, per_device_batch_size=args.batch_size)
    results = runner.evaluate()

    print("Evaluation Results:", results)

if __name__ == "__main__":
    main()
