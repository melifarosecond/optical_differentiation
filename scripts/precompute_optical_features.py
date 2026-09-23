import argparse
import os
import torch
from svetlanna import Wavefront
from optical_differentiation.svetlanna_data import get_wavefront_datasets, cache_tag, DEVICE, Nx
from optical_differentiation.svetlanna_model import SvetlannaOpticalFrontendMultiChannel14


@torch.no_grad()
def precompute(dataset, frontend, batch_size, split_name):
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    feats, labels = [], []
    total = len(dataset)
    done = 0
    for X, y in loader:
        out = frontend(Wavefront(X.to(DEVICE)))
        feats.append(out.cpu())
        labels.append(y)
        done += X.shape[0]
        if done % (batch_size * 50) == 0 or done == total:
            print(f'  {split_name}: {done}/{total}', flush=True)
    return (torch.cat(feats), torch.cat(labels))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--cache-dir', type=str, default='cache')
    parser.add_argument('--directions', type=str, nargs='+', default=['dx', 'dy', 'd45', 'd135'])
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    directions = tuple(args.directions)
    tag = cache_tag(directions)

    print(f'Device: {DEVICE}, сетка {Nx}x{Nx}, каналы {directions}')
    print(f'Тег кэша: {tag}\n')

    train_data, test_data = get_wavefront_datasets()
    frontend = SvetlannaOpticalFrontendMultiChannel14(directions).to(DEVICE)
    frontend.eval()

    for name, data in (('train', train_data), ('test', test_data)):
        path = os.path.join(args.cache_dir, f'optical_{name}_{tag}.pt')
        if os.path.exists(path):
            print(f'{name}: уже существует, пропускаю ({path})')
            continue
        feats, labels = precompute(data, frontend, args.batch_size, name)
        torch.save({'features': feats, 'labels': labels, 'directions': directions, 'nx': Nx}, path)
        size_mb = feats.numel() * 4 / 1000000.0
        print(f'{name}: сохранено {tuple(feats.shape)}, {size_mb:.0f} МБ -> {path}\n')

    print('Готово')


if __name__ == '__main__':
    main()
