import argparse
import torch

from optical_differentiation.data import get_dataloaders
from optical_differentiation.models import LinearOnly, Baseline, BaselineSmall
from optical_differentiation.optical import OpticalModel
from optical_differentiation.train import run

MODELS = {
    "linear": LinearOnly,
    "baseline_a": Baseline,
    "baseline_b": BaselineSmall,
    "optical": OpticalModel,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS.keys(), required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    train_loader, test_loader = get_dataloaders(batch_size=args.batch_size)
    model = MODELS[args.model]()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, model: {args.model}")

    run(
        model,
        train_loader,
        test_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        log_dir=f"runs/{args.model}",
    )


if __name__ == "__main__":
    main()