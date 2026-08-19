import torch
import matplotlib.pyplot as plt

from svetlanna import Wavefront
from optical_differentiation.svetlanna_data import get_wavefront_datasets
from optical_differentiation.svetlanna_model import SvetlannaOpticalFrontend


def main():
    _, test_data = get_wavefront_datasets()
    model = SvetlannaOpticalFrontend() 
    model.eval()

    n_examples = 4
    fig, axes = plt.subplots(2, n_examples, figsize=(12, 6))

    for i in range(n_examples):
        wavefront, label = test_data[i]

        with torch.no_grad():
            out = model(Wavefront(wavefront))

        axes[0, i].imshow(wavefront.intensity, cmap="hot")
        axes[0, i].set_title(f"Вход (класс {label})")
        axes[0, i].axis("off")

        axes[1, i].imshow(out.intensity, cmap="hot")
        axes[1, i].set_title("После метаповерхности (d/dx)")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()