import torch
import matplotlib.pyplot as plt
from svetlanna import Wavefront
from optical_differentiation.svetlanna_data import get_wavefront_datasets, DEVICE
from optical_differentiation.svetlanna_model import SvetlannaOpticalFrontend, IdealOpticalFrontend, FFTOpticalFrontend

def compare(name_a, out_a, name_b, out_b):
    diff = (out_a - out_b).abs()
    max_err = diff.max().item()
    mean_err = diff.mean().item()
    rel_err = (diff.sum() / (out_b.abs().sum() + 1e-12)).item() * 100
    print(f'  {name_a} vs {name_b}:')
    print(f'    max|ошибка|  = {max_err:.3e}')
    print(f'    mean|ошибка| = {mean_err:.3e}')
    print(f'    относительная ошибка (L1) = {rel_err:.4f}%')
    return diff

def main():
    print(f'Device: {DEVICE}')
    _, test_data = get_wavefront_datasets()
    physical = SvetlannaOpticalFrontend().to(DEVICE)
    ideal = IdealOpticalFrontend().to(DEVICE)
    fft_only = FFTOpticalFrontend().to(DEVICE)
    physical.eval()
    ideal.eval()
    fft_only.eval()
    n_examples = 20
    stats = {'physical_vs_ideal': [], 'ideal_vs_fft': [], 'physical_vs_fft': []}
    last_diffs = {}
    for i in range(n_examples):
        wavefront, _ = test_data[i]
        wavefront = wavefront.to(DEVICE)
        with torch.no_grad():
            out_physical = physical(Wavefront(wavefront)).intensity.cpu()
            out_ideal = ideal(Wavefront(wavefront)).intensity.cpu()
            out_fft = fft_only(Wavefront(wavefront)).intensity.cpu()
        d1 = (out_physical - out_ideal).abs()
        d2 = (out_ideal - out_fft).abs()
        d3 = (out_physical - out_fft).abs()
        stats['physical_vs_ideal'].append(d1.mean().item())
        stats['ideal_vs_fft'].append(d2.mean().item())
        stats['physical_vs_fft'].append(d3.mean().item())
        last_diffs = {'physical_vs_ideal': d1, 'ideal_vs_fft': d2, 'physical_vs_fft': d3}
    print(f'Усреднено по {n_examples} примерам (mean|ошибка| по всей картинке):\n')
    print(f'1) Physical (SLM+Aperture) vs Ideal (4f + элемент):')
    print(f'   -> цена разложения на фазу+амплитуду')
    print(f'   среднее mean|ошибка| = {sum(stats['physical_vs_ideal']) / n_examples:.3e}\n')
    print(f'2) Ideal (4f + элемент) vs чистый FFT (без линз):')
    print(f'   -> цена физической реализации Фурье-преобразования (апертура, zpASM)')
    print(f'   среднее mean|ошибка| = {sum(stats['ideal_vs_fft']) / n_examples:.3e}\n')
    print(f'3) Physical (SLM+Aperture) vs чистый FFT (суммарная цена реализации):')
    print(f'   среднее mean|ошибка| = {sum(stats['physical_vs_fft']) / n_examples:.3e}\n')
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles = ['Physical vs Ideal\n(цена фазы+амплитуды)', 'Ideal vs FFT\n(цена физической реализации Фурье)', 'Physical vs FFT\n(суммарная цена)']
    keys = ['physical_vs_ideal', 'ideal_vs_fft', 'physical_vs_fft']
    for ax, title, key in zip(axes, titles, keys):
        im = ax.imshow(last_diffs[key], cmap='inferno')
        ax.set_title(title, fontsize=10)
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig('three_levels_diff.png', dpi=150, bbox_inches='tight')
    print('Сохранено в three_levels_diff.png')
if __name__ == '__main__':
    main()
