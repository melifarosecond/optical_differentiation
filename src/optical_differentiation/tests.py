import torch
from optical_differentiation.train import evaluate
import torch.nn as nn

CHANNEL_NAMES = ["dx", "dy", "laplacian", "blur"]


def zero_channel_hook(channel_idx):
    def hook(module, input, output):
        output = output.clone()         
        output[:, channel_idx, :, :] = 0
        return output
    return hook


def ablate_channels(model, test_loader, device):
    criterion = nn.CrossEntropyLoss()
    model.eval()

    base_loss, base_acc = evaluate(model, test_loader, criterion, device)
    print(f"Все каналы: test_acc={base_acc:.4f}")

    results = {"all_channels": base_acc}

    for idx, name in enumerate(CHANNEL_NAMES):
        handle = model.frontend.pool.register_forward_hook(zero_channel_hook(idx))
        _, acc = evaluate(model, test_loader, criterion, device)
        handle.remove()  

        drop = base_acc - acc
        results[name] = acc
        print(f"Обнулён '{name}': test_acc={acc:.4f}  (падение: {drop:+.4f})")

    return results

def keep_only_channel(model, test_loader, device):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    results = {}

    for keep_idx, name in enumerate(CHANNEL_NAMES):
        def hook(module, input, output, keep_idx=keep_idx):
            output = output.clone()
            mask = torch.zeros_like(output)
            mask[:, keep_idx, :, :] = 1
            return output * mask

        handle = model.frontend.pool.register_forward_hook(hook)
        _, acc = evaluate(model, test_loader, criterion, device)
        handle.remove()

        results[name] = acc
        print(f"Оставлен только '{name}': test_acc={acc:.4f}")

    return results