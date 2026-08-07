import argparse
import random
import numpy as np
import torch

from optical_differentiation.data import get_dataloaders
from optical_differentiation.models import LinearOnly, TwoLinear, Baseline, BaselineNew, BaselineNewSquare
from optical_differentiation.optical import OpticalModel
from optical_differentiation.train import run
from optical_differentiation.tests import ablate_channels, keep_only_channel

MODELS = {
    "linear": LinearOnly,
    "two_linear": TwoLinear,
    "baseline_a": Baseline,
    "baseline_new": BaselineNew,
    "baseline_new_square": BaselineNewSquare,
    "optical": OpticalModel,
}


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS.keys(), required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--ablate", action="store_true")
    args = parser.parse_args()

    if args.seed is not None:
        set_seed(args.seed)

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

    if args.model == "optical" and args.ablate:
        print("\n--- Ablation: обнуление одного канала ---")
        ablate_channels(model, test_loader, device)
        print("\n--- Ablation: оставлен только один канал ---")
        keep_only_channel(model, test_loader, device)


if __name__ == "__main__":
    main()