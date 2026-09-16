import torch
from svetlanna import Wavefront


def train_loop(dataloader, model, loss_fn, optimizer, batch_size, device):
    size = len(dataloader.dataset)
    model.train()

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(Wavefront(X))
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            current = batch * batch_size + len(X)
            print(f"loss: {loss.item():>7f}  [{current:>5d}/{size:>5d}]")


@torch.no_grad()
def test_loop(dataloader, model, loss_fn, device):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0.0, 0

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        pred = model(Wavefront(X))
        test_loss += loss_fn(pred, y).item()
        correct += (pred.argmax(1) == y).float().sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f}")
    return correct, test_loss
