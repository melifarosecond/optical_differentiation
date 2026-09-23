import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from svetlanna import Wavefront
from optical_differentiation.svetlanna_data import get_wavefront_datasets, DEVICE
from optical_differentiation.svetlanna_model import SvetlannaOpticalFrontend, IdealOpticalFrontend, FFTOpticalFrontend

def downsample_for_display(intensity, target=512):
    if intensity.shape[-1] <= target:
        return intensity
    x = intensity.unsqueeze(0).unsqueeze(0)
    x = F.adaptive_avg_pool2d(x, (target, target))
    return x.squeeze(0).squeeze(0)

def robust_normalize(intensity, percentile=99.9):
    flat = intensity.flatten()
    vmax = torch.quantile(flat, percentile / 100.0).item()
    vmax = max(vmax, 1e-30)
    return (intensity / vmax).clamp(0, 1)

def thicken(intensity, fraction=0.008):
    k = max(3, int(intensity.shape[-1] * fraction) | 1)
    x = intensity.unsqueeze(0).unsqueeze(0)
    x = F.max_pool2d(x, kernel_size=k, stride=1, padding=k // 2)
    return x.squeeze(0).squeeze(0)

def flip_4f_output(intensity):
    return torch.flip(intensity, dims=[-2, -1])

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
    col_titles = ['Вход', 'Physical\n(SLM+Aperture, развёрнуто\n180° для сравнения)', 'Ideal\n(4f + элемент, развёрнуто\n180° для сравнения)', 'FFT\n(без линз)']
    for i in range(n_examples):
        wavefront, label = test_data[i]
        wavefront = wavefront.to(DEVICE)
        with torch.no_grad():
            out_physical = physical(Wavefront(wavefront)).intensity.cpu()
            out_ideal = ideal(Wavefront(wavefront)).intensity.cpu()
            out_fft = fft_only(Wavefront(wavefront)).intensity.cpu()
        out_physical = flip_4f_output(out_physical)
        out_ideal = flip_4f_output(out_ideal)
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
