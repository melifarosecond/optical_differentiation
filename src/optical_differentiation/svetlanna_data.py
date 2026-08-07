import torch
import svetlanna as sv
from svetlanna.units import ureg
from torchvision.datasets import FashionMNIST
import torchvision.transforms as transforms
from svetlanna.transforms import ToWavefront

WAVELENGTH = 630 * ureg.nm  

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


def create_segment_mask(x: int, y: int, d: int):
    res = torch.zeros((Ny, Nx))
    res[
        (Ny - d) // 2 + y: (Ny + d) // 2 + y,
        (Nx - d) // 2 + x: (Nx + d) // 2 + x,
    ] = 1.0
    return res


def build_detector_masks():
    d = 16

    masks = torch.stack([
        create_segment_mask(-int(2.7 * d), -int(2.7 * d), d),
        create_segment_mask(0, -int(2.7 * d), d),
        create_segment_mask(int(2.7 * d), -int(2.7 * d), d),
        create_segment_mask(-int(3 * d), 0, d),
        create_segment_mask(-int(1 * d), 0, d),
        create_segment_mask(int(1 * d), 0, d),
        create_segment_mask(int(3 * d), 0, d),
        create_segment_mask(-int(2.7 * d), int(2.7 * d), d),
        create_segment_mask(0, int(2.7 * d), d),
        create_segment_mask(int(2.7 * d), int(2.7 * d), d),
    ], dim=-1)
    return masks


def build_x_derivative_phase_mask():
    x_axis = torch.linspace(-GRID_SIZE / 2, GRID_SIZE / 2, Nx)
    kx = (2 * torch.pi / (WAVELENGTH * FOCAL_LENGTH)) * x_axis

    transfer_function = 1j * kx  
    phase_mask_1d = torch.angle(transfer_function) % (2 * torch.pi)

    phase_mask = phase_mask_1d.unsqueeze(0).expand(Ny, Nx).clone()

    eps = 1e-3
    phase_mask = phase_mask.clamp(eps, 2 * torch.pi - eps)

    return phase_mask