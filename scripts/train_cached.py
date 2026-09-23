import argparse
import os
import random
import numpy as np
import torch
from torch import nn
from optical_differentiation.svetlanna_data import cache_tag, DEVICE
from optical_differentiation.svetlanna_model import ElectronicHead


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_cached(cache_dir, split, tag):
    path = os.path.join(cache_dir, f'optical_{split}_{tag}.pt')
    if not os.path.exists(path):
        raise FileNotFoundError(f'Кэш не найден: {path}\nСначала запустите scripts/precompute_optical_features.py')
    blob = torch.load(path)
    return (blob['features'], blob['labels'])


def build_optimizer(name, params, lr):
    name = name.lower()
    if name == 'adam':
        return torch.optim.Adam(params, lr=lr)
    if name == 'sgd':
        return torch.optim.SGD(params, lr=lr)
    if name == 'sgd_momentum':
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    if name == 'rmsprop':
        return torch.optim.RMSprop(params, lr=lr)
    raise ValueError(f'Неизвестный оптимизатор: {name}')


def build_scheduler(name, optimizer, epochs, steps_per_epoch, lr):
    name = (name or 'none').lower()
    if name == 'none':
        return (None, False)
    if name == 'step':
        return (torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.5), False)
    if name == 'cosine':
        return (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs), False)
    if name == 'onecycle':
        return (torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=steps_per_epoch, epochs=epochs), True)
    raise ValueError(f'Неизвестный scheduler: {name}')


def train_one_epoch(model, loader, optimizer, criterion, scheduler, per_step):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        if per_step and scheduler is not None:
            scheduler.step()
        total_loss += loss.item() * X.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += X.size(0)
    return (total_loss / total, correct / total)


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        out = model(X)
        loss = criterion(out, y)
        total_loss += loss.item() * X.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += X.size(0)
    return (total_loss / total, correct / total)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd', 'sgd_momentum', 'rmsprop'])
    parser.add_argument('--scheduler', type=str, default='none', choices=['none', 'step', 'cosine', 'onecycle'])
    parser.add_argument('--cache-dir', type=str, default='cache')
    parser.add_argument('--directions', type=str, nargs='+', default=['dx', 'dy', 'd45', 'd135'])
    parser.add_argument('--no-batchnorm', action='store_true')
    parser.add_argument('--log-dir', type=str, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        set_seed(args.seed)

    tag = cache_tag(tuple(args.directions))
    train_X, train_y = load_cached(args.cache_dir, 'train', tag)
    test_X, test_y = load_cached(args.cache_dir, 'test', tag)
    print(f'Device: {DEVICE}, кэш {tag}')
    print(f'train {tuple(train_X.shape)}, test {tuple(test_X.shape)}')

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_X, train_y), batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_X, test_y), batch_size=512, shuffle=False)

    model = ElectronicHead(n_channels=train_X.shape[1], use_batchnorm=not args.no_batchnorm).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(args.optimizer, model.parameters(), args.lr)
    scheduler, per_step = build_scheduler(args.scheduler, optimizer, args.epochs, len(train_loader), args.lr)

    writer = None
    if args.log_dir:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(args.log_dir)

    best_acc = 0.0
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, scheduler, per_step)
        test_loss, test_acc = evaluate(model, test_loader, criterion)
        if scheduler is not None and not per_step:
            scheduler.step()
        best_acc = max(best_acc, test_acc)
        print(f'epoch {epoch + 1}/{args.epochs}  train_loss={train_loss:.4f} train_acc={train_acc:.4f}  test_loss={test_loss:.4f} test_acc={test_acc:.4f}')
        if writer:
            writer.add_scalars('loss', {'train': train_loss, 'test': test_loss}, epoch)
            writer.add_scalars('acc', {'train': train_acc, 'test': test_acc}, epoch)

    if writer:
        writer.close()
    print(f'\nЛучший test_acc: {best_acc:.4f}')


if __name__ == '__main__':
    main()
