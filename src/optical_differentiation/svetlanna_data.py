import torch
import svetlanna as sv
from svetlanna.units import ureg
from torchvision.datasets import FashionMNIST
import torchvision.transforms as transforms
from svetlanna.transforms import ToWavefront

WAVELENGTH = 632 * ureg.nm
Nx, Ny = 200, 200
GRID_SIZE = 8 * ureg.mm
FOCAL_LENGTH = 5 * ureg.cm

SIM_PARAMS = sv.SimulationParameters(
    x=torch.linspace(-GRID_SIZE / 2, GRID_SIZE / 2, Nx),
    y=torch.linspace(-GRID_SIZE / 2, GRID_SIZE / 2, Ny),
    wavelength=WAVELENGTH,
)


def get_wavefront_datasets(data_dir: str = "data"):
    to_wavefront_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size=(100, 100), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.Pad(padding=(50, 50, 50, 50), fill=0),
        ToWavefront(modulation_type="amp"),
    ])

    train_data = FashionMNIST(root=data_dir, train=True, download=True, transform=to_wavefront_transform)
    test_data = FashionMNIST(root=data_dir, train=False, download=True, transform=to_wavefront_transform)
    return train_data, test_data


def build_x_derivative_phase_mask() -> torch.Tensor:
    eps = 1e-3  # не даём значениям попасть точно на границу [0, 2*pi]
    x_axis = torch.linspace(-GRID_SIZE / 2, GRID_SIZE / 2, Nx)
    kx = (2 * torch.pi / (WAVELENGTH * FOCAL_LENGTH)) * x_axis

    transfer_function = 1j * kx
    phase_mask_1d = torch.angle(transfer_function) % (2 * torch.pi)
    phase_mask_1d = phase_mask_1d.clamp(eps, 2 * torch.pi - eps)

    phase_mask = phase_mask_1d.unsqueeze(0).expand(Ny, Nx).clone()
    return phase_mask


def build_x_derivative_amplitude_mask() -> torch.Tensor:
    x_axis = torch.linspace(-GRID_SIZE / 2, GRID_SIZE / 2, Nx)
    kx = (2 * torch.pi / (WAVELENGTH * FOCAL_LENGTH)) * x_axis

    amplitude_1d = kx.abs() / kx.abs().max()
    amplitude = amplitude_1d.unsqueeze(0).expand(Ny, Nx).clone()
    return amplitude