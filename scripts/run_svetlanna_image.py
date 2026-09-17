import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from svetlanna import Wavefront
from optical_differentiation.svetlanna_data import get_wavefront_datasets, DEVICE
from optical_differentiation.svetlanna_model import SvetlannaOpticalFrontend, IdealOpticalFrontend, FFTOpticalFrontend

def robust_normalize(intensity, percentile=99.9):
    flat = intensity.flatten()
    vmax = torch.quantile(flat, percentile / 100.0).item()
    vmax = max(vmax, 1e-30)
    return (intensity / vmax).clamp(0, 1)

def thicken(intensity, kernel_size=3):
    x = intensity.unsqueeze(0).unsqueeze(0)
    x = F.max_pool2d(x, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    return x.squeeze(0).squeeze(0)

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
    apply_thicken = True
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
            display_img = robust_normalize(img)
            if apply_thicken and j > 0:
                display_img = thicken(display_img)
            ax.imshow(display_img, cmap='hot', vmin=0, vmax=1)
            ax.axis('off')
            if i == 0:
                ax.set_title(title, fontsize=11)
            if j == 0:
                ax.text(-0.15, 0.5, f'класс {label}', transform=ax.transAxes, rotation=90, va='center', ha='center', fontsize=10)
    plt.tight_layout()
    plt.savefig('svetlanna_comparison.png', dpi=150, bbox_inches='tight')
    print('Сохранено в svetlanna_comparison.png')
if __name__ == '__main__':
    main()
