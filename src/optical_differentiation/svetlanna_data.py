import math
import torch
import svetlanna as sv
from svetlanna.units import ureg
from torchvision.datasets import FashionMNIST
import torchvision.transforms as transforms
from svetlanna.transforms import ToWavefront

WAVELENGTH = 632.8 * ureg.nm
Nx, Ny = 2048, 2048
GRID_SIZE = 8 * ureg.mm
FOCAL_LENGTH = 5 * ureg.cm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SIM_PARAMS = sv.SimulationParameters(x=torch.linspace(-GRID_SIZE / 2, GRID_SIZE / 2, Nx), y=torch.linspace(-GRID_SIZE / 2, GRID_SIZE / 2, Ny), wavelength=WAVELENGTH)

CHANNEL_DIRECTIONS = {'dx': 0.0, 'dy': 90.0, 'd45': 45.0, 'd135': 135.0}

OUTPUT_SIZE = 14


def cache_tag(directions):
    return f"nx{Nx}_{'-'.join(directions)}_out{OUTPUT_SIZE}"


def get_wavefront_datasets(data_dir='data'):
    resize_size = Nx // 2
    pad_size = (Nx - resize_size) // 2
    to_wavefront_transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(size=(resize_size, resize_size), interpolation=transforms.InterpolationMode.NEAREST), transforms.Pad(padding=(pad_size, pad_size, pad_size, pad_size), fill=0), ToWavefront(modulation_type='amp')])
    train_data = FashionMNIST(root=data_dir, train=True, download=True, transform=to_wavefront_transform)
    test_data = FashionMNIST(root=data_dir, train=False, download=True, transform=to_wavefront_transform)
    return (train_data, test_data)


def build_directional_transfer_function(theta_deg):
    x_axis = torch.linspace(-GRID_SIZE / 2, GRID_SIZE / 2, Nx)
    y_axis = torch.linspace(-GRID_SIZE / 2, GRID_SIZE / 2, Ny)
    k_scale = 2 * torch.pi / (WAVELENGTH * FOCAL_LENGTH)
    kx = k_scale * x_axis
    ky = k_scale * y_axis
    kx_grid = kx.unsqueeze(0).expand(Ny, Nx)
    ky_grid = ky.unsqueeze(1).expand(Ny, Nx)
    theta = math.radians(theta_deg)
    k_theta = kx_grid * math.cos(theta) + ky_grid * math.sin(theta)
    return (1j * k_theta).to(torch.complex64).clone()


def build_ideal_normalized_transfer_function(theta_deg):
    tf = build_directional_transfer_function(theta_deg)
    return tf / tf.abs().max()


def _grid_spacing_meters():
    spacing_quantity = GRID_SIZE / Nx
    try:
        return float(spacing_quantity.to(ureg.m).magnitude)
    except AttributeError:
        return float(spacing_quantity)


def build_directional_fft_kernel(theta_deg):
    dx_spacing = _grid_spacing_meters()
    kx = 2 * torch.pi * torch.fft.fftfreq(Nx, d=dx_spacing)
    ky = 2 * torch.pi * torch.fft.fftfreq(Ny, d=dx_spacing)
    kx_grid = kx.unsqueeze(0).expand(Ny, Nx)
    ky_grid = ky.unsqueeze(1).expand(Ny, Nx)
    theta = math.radians(theta_deg)
    k_theta = kx_grid * math.cos(theta) + ky_grid * math.sin(theta)
    kernel = (1j * k_theta).to(torch.complex64)
    return (kernel / kernel.abs().max()).clone()


def derive_phase_amplitude_masks(transfer_function, eps=0.001):
    amplitude = transfer_function.abs()
    amplitude = amplitude / amplitude.max()
    phase = torch.angle(transfer_function) % (2 * torch.pi)
    phase = phase.clamp(eps, 2 * torch.pi - eps)
    return (phase, amplitude)


def build_x_derivative_phase_mask():
    tf = build_directional_transfer_function(CHANNEL_DIRECTIONS['dx'])
    phase, _ = derive_phase_amplitude_masks(tf)
    return phase


def build_x_derivative_amplitude_mask():
    tf = build_directional_transfer_function(CHANNEL_DIRECTIONS['dx'])
    _, amplitude = derive_phase_amplitude_masks(tf)
    return amplitude
