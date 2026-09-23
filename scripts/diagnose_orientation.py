import torch
from svetlanna import Wavefront
from optical_differentiation.svetlanna_data import get_wavefront_datasets, DEVICE
from optical_differentiation.svetlanna_model import SvetlannaOpticalFrontend, FFTOpticalFrontend


def rel_error(a, b):
    diff = (a - b).abs()
    return (diff.sum() / (b.abs().sum() + 1e-12)).item() * 100


def main():
    print(f'Device: {DEVICE}')
    _, test_data = get_wavefront_datasets()

    physical = SvetlannaOpticalFrontend().to(DEVICE)
    fft_only = FFTOpticalFrontend().to(DEVICE)
    physical.eval()
    fft_only.eval()

    variants = {
        'без изменений': lambda x: x,
        'flip по вертикали (dim=-2)': lambda x: torch.flip(x, dims=[-2]),
        'flip по горизонтали (dim=-1)': lambda x: torch.flip(x, dims=[-1]),
        'flip по обеим осям (180°)': lambda x: torch.flip(x, dims=[-2, -1]),
        'roll N/2': lambda x: torch.roll(x, (x.shape[-2] // 2, x.shape[-1] // 2), dims=(-2, -1)),
        'flip + roll N/2': lambda x: torch.roll(torch.flip(x, [-2, -1]), (x.shape[-2] // 2, x.shape[-1] // 2), dims=(-2, -1)),
    }

    n_examples = 10
    totals = {name: 0.0 for name in variants}

    for i in range(n_examples):
        wavefront, _ = test_data[i]
        wavefront = wavefront.to(DEVICE)
        with torch.no_grad():
            out_physical = physical(Wavefront(wavefront)).intensity.cpu()
            out_fft = fft_only(Wavefront(wavefront)).intensity.cpu()

        for name, transform in variants.items():
            totals[name] += rel_error(transform(out_physical), out_fft)

    print(f'Physical vs FFT, усреднено по {n_examples} примерам, разные варианты ориентации Physical:\n')
    for name, total in totals.items():
        avg = total / n_examples
        print(f'  {name}: относительная ошибка = {avg:.4f}%')

    best = min(totals, key=totals.get)
    print(f'\nЛучшее совпадение: "{best}"')
    if best == 'без изменений':
        print('Реального переворота нет — Physical и FFT ориентированы одинаково.')
    else:
        print(f'Обнаружен систематический сдвиг ориентации: {best}')


if __name__ == '__main__':
    main()
