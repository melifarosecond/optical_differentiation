import argparse
import torch
from torch import nn

from optical_differentiation.svetlanna_data import get_wavefront_datasets, DEVICE
from optical_differentiation.svetlanna_model import (
    SvetlannaOpticalModelEconomical,
    SvetlannaOpticalModelMultiChannel,
)
from optical_differentiation.svetlanna_train import train_loop, test_loop

MODELS = {
    "single": SvetlannaOpticalModelEconomical,   # 1 физический канал (d/dx)
    "multi": SvetlannaOpticalModelMultiChannel,  # 4 физических канала (dx/dy/d45/d135)
}

# Многоканальная версия считает заднюю часть системы на B*4 "виртуальных"
# примерах за раз — по умолчанию батч меньше, чтобы не упереться в память.
DEFAULT_BATCH_SIZE = {"single": 8, "multi": 2}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS.keys(), default="single")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=None,
                         help="Если не задан — берётся разумное значение по умолчанию для модели")
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    batch_size = args.batch_size or DEFAULT_BATCH_SIZE[args.model]

    train_data, test_data = get_wavefront_datasets()

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=max(batch_size, 32), shuffle=False)

    model = MODELS[args.model]().to(DEVICE)

    # Проверка перед обучением: убедиться, что оптика зафиксирована
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    print(f"Device: {DEVICE}")
    print(f"Модель: {args.model}, batch_size={batch_size}")
    print("Обучаемые параметры:", trainable)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    print("\nДо обучения:")
    test_loop(test_dataloader, model, loss_fn, DEVICE)

    for t in range(args.epochs):
        print(f"\nEpoch {t + 1}/{args.epochs}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer, batch_size, DEVICE)
        test_loop(test_dataloader, model, loss_fn, DEVICE)

    print("Готово!")


if __name__ == "__main__":
    main()
