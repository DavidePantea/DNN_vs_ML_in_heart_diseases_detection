import argparse

from src.models.model_a import train as train_a
from src.models.model_b import train as train_b

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Run training")
    parser.add_argument("--model", choices=["a", "b"], help="Which model to train")
    parser.add_argument("--epochs", type=int, default=10)

    args = parser.parse_args()

    if not args.train:
        print("Container started (no training). Use --train --model a|b to train.")
        return

    if args.model is None:
        parser.error("--model is required when using --train")

    ensure_dataset()

    if args.model == "a":
        train_a(epochs=args.epochs)
    else:
        train_b(epochs=args.epochs)


def ensure_dataset():
    from scripts.get_dataset import main as download_main
    download_main()

if __name__ == "__main__":
    ensure_dataset()
    main()
