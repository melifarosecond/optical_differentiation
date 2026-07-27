from torchvision import datasets, transforms
from torch.utils.data import DataLoader


DEFAULT_DATA_DIR = "data"


def get_dataloaders(batch_size: int = 128, data_dir: str = DEFAULT_DATA_DIR):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])

    train_set = datasets.FashionMNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_set = datasets.FashionMNIST(
        root=data_dir, train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader