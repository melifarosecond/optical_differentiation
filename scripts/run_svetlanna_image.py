import torch 
import matplotlib.pyplot as plt

from svetlanna import Wavefront 
from optical_differentiation.svetlanna_data import get_wavefront_datasets 
from optical_differentiation.svetlanna_model import ( 
    SvetlannaOpticalFrontend, 
    IdealOpticalFrontend, 
    FFTOpticalFrontend, 
)
def main(): 
    _, test_data = get_wavefront_datasets()

    physical = SvetlannaOpticalFrontend()
    ideal = IdealOpticalFrontend()
    fft_only = FFTOpticalFrontend()
    physical.eval()
    ideal.eval()
    fft_only.eval()

    n_examples = 3
    fig, axes = plt.subplots(n_examples, 4, figsize=(14, 3.2 * n_examples))

    col_titles = ["Вход", "Physical\n(SLM+Aperture)", "Ideal\n(4f + элемент)", "FFT\n(без линз)"]

    for i in range(n_examples):
        wavefront, label = test_data[i]

        with torch.no_grad():
            out_physical = physical(Wavefront(wavefront)).intensity
            out_ideal = ideal(Wavefront(wavefront)).intensity
            out_fft = fft_only(Wavefront(wavefront)).intensity

        images = [wavefront.intensity, out_physical, out_ideal, out_fft]

        for j, (img, title) in enumerate(zip(images, col_titles)):
            ax = axes[i, j]
            ax.imshow(img, cmap="hot")
            ax.axis("off")
            if i == 0:
                ax.set_title(title, fontsize=11)
            if j == 0:
                ax.text(-0.15, 0.5, 
                        f"класс {label}", transform=ax.transAxes,
                    rotation=90, 
                    va="center", ha="center", fontsize=10)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__": 
    main()