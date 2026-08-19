import torch
import torch.nn as nn


def build_optimizer(name: str, params, lr: float):
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    elif name == "sgd":
        return torch.optim.SGD(params, lr=lr)
    elif name == "sgd_momentum":
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    elif name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr)
    else:
        raise ValueError(f"Неизвестный оптимизатор: {name}")


def build_scheduler(name: str, optimizer, epochs: int, steps_per_epoch: int, lr: float):
    name = (name or "none").lower()
    if name == "none":
        return None, False
    elif name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.5), False
    elif name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs), False
    elif name == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr, steps_per_epoch=steps_per_epoch, epochs=epochs
        ), True
    else:
        raise ValueError(f"Неизвестный scheduler: {name}")


def train_one_epoch(model, loader, optimizer, criterion, device, scheduler=None, step_scheduler_per_batch=False):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if step_scheduler_per_batch and scheduler is not None:
            scheduler.step() 

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total


def run(model, train_loader, test_loader, epochs=10, lr=1e-3, device=None, log_dir=None,
        optimizer_name="adam", scheduler_name="none"):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = build_optimizer(optimizer_name, trainable_params, lr)

    scheduler, per_step = build_scheduler(scheduler_name, optimizer, epochs, len(train_loader), lr)

    writer = None
    if log_dir:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir)

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            scheduler=scheduler, step_scheduler_per_batch=per_step,
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"epoch {epoch+1}/{epochs} lr={current_lr:.2e} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}")

        if scheduler is not None and not per_step:
            scheduler.step() # StepLR/CosineAnnealingLR — раз в эпоху

        if writer:
            writer.add_scalars("loss", {"train": train_loss, "test": test_loss}, epoch)
            writer.add_scalars("acc", {"train": train_acc, "test": test_acc}, epoch)
            writer.add_scalar("lr", current_lr, epoch)

    if writer:
        writer.close()
    return model