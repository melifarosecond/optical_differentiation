import torch
import torch.nn as nn


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

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


def run(model, train_loader, test_loader, epochs=10, lr=1e-3, device=None, log_dir=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(trainable_params, lr=lr)

    writer = None
    if log_dir:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir)

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        print(f"epoch {epoch + 1}/{epochs}  "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
              f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}")

        if writer:
            writer.add_scalars("loss", {"train": train_loss, "test": test_loss}, epoch)
            writer.add_scalars("acc", {"train": train_acc, "test": test_acc}, epoch)

    if writer:
        writer.close()
    return model