import argparse
import random
import numpy as np
import torch

from optical_differentiation.data import get_dataloaders
from optical_differentiation.models import LinearOnly, TwoLinear, ThreeLinear, ConvOneLayer, Baseline, BaselineNew, BaselineNewSquare
from optical_differentiation.optical import OpticalModel, OpticalModelDiagonal, OpticalDiagonalOneLayer
from optical_differentiation.train import run
from optical_differentiation.ablation import ablate_channels, keep_only_channel

MODELS = {
    "linear": LinearOnly,
    "two_linear": TwoLinear,
    "three_linear": ThreeLinear,
    "conv_one_layer": ConvOneLayer,
    "baseline_a": Baseline,
    "baseline_new": BaselineNew,
    "baseline_new_square": BaselineNewSquare,
    "optical": OpticalModel,
    "optical_diagonal": OpticalModelDiagonal,
    "optical_diagonal_one_layer": OpticalDiagonalOneLayer,
}

CHANNEL_NAMES = {
    "optical": ["dx", "dy", "laplacian", "blur"],
    "optical_diagonal": ["dx", "dy", "d45", "d135"],
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
    parser.add_argument("--optimizer", type=str, default="adam",
        choices=["adam", "sgd", "sgd_momentum", "rmsprop"])
    parser.add_argument("--scheduler", type=str, default="none",
        choices=["none", "step", "cosine", "onecycle"])
    args = parser.parse_args()

    if args.seed is not None:
        set_seed(args.seed)

    train_loader, test_loader = get_dataloaders(batch_size=args.batch_size)
    model = MODELS[args.model]()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, model: {args.model}, "
        f"optimizer: {args.optimizer}, scheduler: {args.scheduler}, "
        f"batch_size: {args.batch_size}")

    run_name = f"{args.model}_{args.optimizer}_{args.scheduler}_bs{args.batch_size}"

    model = run(
        model,
        train_loader,
        test_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        log_dir=f"runs/{run_name}",
        optimizer_name=args.optimizer,
        scheduler_name=args.scheduler,
    )
    if args.model in CHANNEL_NAMES and args.ablate:
        names = CHANNEL_NAMES[args.model]
        print("\n--- Ablation: обнуление одного канала ---")
        ablate_channels(model, test_loader, device, names)
        print("\n--- Ablation: оставлен только один канал ---")
        keep_only_channel(model, test_loader, device, names)


if __name__ == "__main__":
    main()