import torch
import torch.nn as nn
from optical_differentiation.train import evaluate


def zero_channel_hook(channel_idx):
    def hook(module, input, output):
        output = output.clone()
        output[:, channel_idx, :, :] = 0
        return output
    return hook


def keep_only_channel_hook(channel_idx):
    def hook(module, input, output):
        mask = torch.zeros_like(output)
        mask[:, channel_idx, :, :] = 1
        return output * mask
    return hook


def ablate_channels(model, test_loader, device, channel_names: list[str]):
    criterion = nn.CrossEntropyLoss()
    model.eval()

    base_loss, base_acc = evaluate(model, test_loader, criterion, device)
    print(f"Все каналы: test_acc={base_acc:.4f}")

    results = {"all_channels": base_acc}

    for idx, name in enumerate(channel_names):
        handle = model.frontend.pool.register_forward_hook(zero_channel_hook(idx))
        _, acc = evaluate(model, test_loader, criterion, device)
        handle.remove()

        drop = base_acc - acc
        results[name] = acc
        print(f"Обнулён '{name}': test_acc={acc:.4f}  (падение: {drop:+.4f})")

    return results


def keep_only_channel(model, test_loader, device, channel_names: list[str]):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    results = {}

    for idx, name in enumerate(channel_names):
        handle = model.frontend.pool.register_forward_hook(keep_only_channel_hook(idx))
        _, acc = evaluate(model, test_loader, criterion, device)
        handle.remove()

        results[name] = acc
        print(f"Оставлен только '{name}': test_acc={acc:.4f}")

    return results