import torch
from torch import nn

from optical_differentiation.svetlanna_data import get_wavefront_datasets
from optical_differentiation.svetlanna_model import OpticalClassifier
from optical_differentiation.svetlanna_train import train_loop, test_loop


def main():
    batch_size = 8 
    epochs = 1

    train_data, test_data = get_wavefront_datasets()

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=False)

    model = OpticalClassifier() 
    optimizer = torch.optim.Adam(params=model.parameters(), lr=5e-4)
    loss_fn = nn.CrossEntropyLoss()

    print("До обучения:")
    test_loop(test_dataloader, model, loss_fn)

    for t in range(epochs):
        print(f"\nEpoch {t + 1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer, batch_size)
        test_loop(test_dataloader, model, loss_fn)

    print("Готово!")


if __name__ == "__main__":
    main()