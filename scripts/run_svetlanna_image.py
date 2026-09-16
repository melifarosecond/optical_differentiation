import torch
import matplotlib.pyplot as plt
from svetlanna import Wavefront
from optical_differentiation.svetlanna_data import get_wavefront_datasets, DEVICE
from optical_differentiation.svetlanna_model import SvetlannaOpticalFrontend, IdealOpticalFrontend, FFTOpticalFrontend

def to_log_display(intensity, decades=6):
    peak = intensity.max()
    normalized = intensity / (peak + 1e-30)
    floor = 10.0 ** (-decades)
    return torch.log10(normalized.clamp(min=floor))

def main():
    print(f'Device: {DEVICE}')
    _, test_data = get_wavefront_datasets()
    physical = SvetlannaOpticalFrontend().to(DEVICE)
    ideal = IdealOpticalFrontend().to(DEVICE)
    fft_only = FFTOpticalFrontend().to(DEVICE)
    physical.eval()
    ideal.eval()
    fft_only.eval()
    n_examples = 3
    decades = 6
    fig, axes = plt.subplots(n_examples, 4, figsize=(14, 3.2 * n_examples))
    col_titles = ['Вход', 'Physical\n(SLM+Aperture)', 'Ideal\n(4f + элемент)', 'FFT\n(без линз)']
    for i in range(n_examples):
        wavefront, label = test_data[i]
        wavefront = wavefront.to(DEVICE)
        with torch.no_grad():
            out_physical = physical(Wavefront(wavefront)).intensity.cpu()
            out_ideal = ideal(Wavefront(wavefront)).intensity.cpu()
            out_fft = fft_only(Wavefront(wavefront)).intensity.cpu()
        images = [wavefront.cpu().intensity, out_physical, out_ideal, out_fft]
        for j, (img, title) in enumerate(zip(images, col_titles)):
            ax = axes[i, j]
            log_img = to_log_display(img, decades=decades)
            im = ax.imshow(log_img, cmap='hot', vmin=-decades, vmax=0)
            ax.axis('off')
            if i == 0:
                ax.set_title(title, fontsize=11)
            if j == 0:
                ax.text(-0.15, 0.5, f'класс {label}', transform=ax.transAxes, rotation=90, va='center', ha='center', fontsize=10)
    cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    cbar.set_label(f'log10(I / I_max), {decades} порядков вниз')
    plt.savefig('svetlanna_comparison.png', dpi=150, bbox_inches='tight')
    print('Сохранено в svetlanna_comparison.png')
if __name__ == '__main__':
    main()
