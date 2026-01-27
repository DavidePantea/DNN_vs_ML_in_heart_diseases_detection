import argparse

from src.models.model_a import train as train_a
from src.models.model_b import train as train_b

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["a", "b"], required=True)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    if args.model == "a":
        train_a(epochs=args.epochs)
    else:
        train_b(epochs=args.epochs)

if __name__ == "__main__":
    main()
