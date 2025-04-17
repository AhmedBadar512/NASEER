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
    parser.add_argument("--max_length", type=int, default=512, help="Maximum input length for the model")
    parser.add_argument("--report_to",
                        type=str,
                        default="tensorboard",
                        help="Comma-separated list: tensorboard,wandb,…")
    parser.add_argument("--logging_dir",
                        type=str,
                        default=None,
                        help="Override default logging directory")
    parser.add_argument("--save_every_n_epochs",
                        type=int,
                        default=None,
                        help="If set, save model every N epochs")
    parser.add_argument("--learning_rate",
                        type=float,
                        default=None,
                        help="Initial learning rate")
    parser.add_argument("--lr_scheduler_type",
                        type=str,
                        default='linear',
                        help="LR scheduler type (linear, cosine, …)")
    parser.add_argument("--warmup_steps",
                        type=int,
                        default=0,
                        help="Number of warmup steps")
    parser.add_argument("--use_naseer",
                        action="store_true",
                        help="Replace HF attention with NASEER layers")
    parser.add_argument("--layer_entangle_method",
                        type=str,
                        default="gated",
                        help="Entanglement for NASEER")
    parser.add_argument("--layer_top_k",
                        type=int,
                        default=None,
                        help="Keep top-k interactions")
    parser.add_argument("--layer_rank",
                        type=int,
                        default=8,
                        help="Rank for low-rank entanglement")
    parser.add_argument("--layer_hidden_size",
                        type=int,
                        default=None,
                        help="Override hidden_size in NASEERLayer")
    parser.add_argument("--layer_num_heads",
                        type=int,
                        default=None,
                        help="Override num_heads in NASEERLayer")
    args = parser.parse_args()

    dst_dir = os.path.join(args.project_dir, args.model.split("/")[-1])
    os.makedirs(dst_dir, exist_ok=True)
    print(f"Saving checkpoints to {dst_dir}")

    runner = HFModelRunner(
        model_name=args.model,
        dataset_path=args.dataset,
        output_dir=dst_dir,
        max_length=args.max_length,  # Input context length
        anli_round=args.anli_round,  # Pass the ANLI round argument
        use_naseer=args.use_naseer,
        entangle_method=args.layer_entangle_method,
        top_k=args.layer_top_k,
        rank=args.layer_rank,
        layer_hidden_size=args.layer_hidden_size,
        layer_num_heads=args.layer_num_heads
    )

    train_metrics = runner.train(
        num_train_epochs=args.epochs,
        per_device_batch_size=args.batch_size,
        report_to=args.report_to.split(",") if args.report_to else None,
        save_every_n_epochs=args.save_every_n_epochs,
        logging_dir=args.logging_dir,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps
    )
    print("Training Metrics:", train_metrics)

    results = runner.eval()

    print("Evaluation Results:", results)

if __name__ == "__main__":
    main()
