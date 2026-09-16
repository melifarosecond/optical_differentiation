import math
import torch
import svetlanna as sv
from svetlanna.units import ureg
from torchvision.datasets import FashionMNIST
import torchvision.transforms as transforms
from svetlanna.transforms import ToWavefront

# --- Параметры симуляции ---
WAVELENGTH = 632.8 * ureg.nm
Nx, Ny = 2048, 2048
GRID_SIZE = 8 * ureg.mm
FOCAL_LENGTH = 5 * ureg.cm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SIM_PARAMS = sv.SimulationParameters(
    x=torch.linspace(-GRID_SIZE / 2, GRID_SIZE / 2, Nx),
    y=torch.linspace(-GRID_SIZE / 2, GRID_SIZE / 2, Ny),
    wavelength=WAVELENGTH,
)

# Направления производных, которые используем как физические каналы.
# Совпадают по смыслу с dx/dy/45°/135° из PyTorch OpticalFrontendDiagonal.
CHANNEL_DIRECTIONS = {"dx": 0.0, "dy": 90.0, "d45": 45.0, "d135": 135.0}


def get_wavefront_datasets(data_dir: str = "data"):
    """Fashion-MNIST -> Wavefront (амплитудная модуляция)."""
    to_wavefront_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size=(100, 100), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.Pad(padding=(50, 50, 50, 50), fill=0),
        ToWavefront(modulation_type="amp"),
    ])

    train_data = FashionMNIST(root=data_dir, train=True, download=True, transform=to_wavefront_transform)
    test_data = FashionMNIST(root=data_dir, train=False, download=True, transform=to_wavefront_transform)
    return train_data, test_data


def build_directional_transfer_function(theta_deg: float) -> torch.Tensor:
    """Точная (не нормированная) передаточная функция направленной
    производной под углом theta для ФИЗИЧЕСКОЙ 4f-системы:
    H(kx,ky) = i * (kx*cos(theta) + ky*sin(theta)).

    theta=0   -> d/dx
    theta=90  -> d/dy
    theta=45  -> производная под 45°
    theta=135 -> производная под 135°

    Координата на плоскости SLM в 4f-системе сама является пространственной
    частотой: k = (2*pi/(wavelength*focal_length)) * координата. Эта формула
    работает ТОЛЬКО когда поле физически прошло через линзу и FreeSpace —
    для прямого FFT (без линз) нужна другая функция, см.
    build_directional_fft_kernel ниже.
    """
    x_axis = torch.linspace(-GRID_SIZE / 2, GRID_SIZE / 2, Nx)
    y_axis = torch.linspace(-GRID_SIZE / 2, GRID_SIZE / 2, Ny)

    k_scale = 2 * torch.pi / (WAVELENGTH * FOCAL_LENGTH)
    kx = k_scale * x_axis  # [Nx]
    ky = k_scale * y_axis  # [Ny]

    kx_grid = kx.unsqueeze(0).expand(Ny, Nx)  # меняется по столбцам (x)
    ky_grid = ky.unsqueeze(1).expand(Ny, Nx)  # меняется по строкам (y)

    theta = math.radians(theta_deg)
    k_theta = kx_grid * math.cos(theta) + ky_grid * math.sin(theta)

    transfer_function = (1j * k_theta).to(torch.complex64)
    return transfer_function.clone()


def build_ideal_normalized_transfer_function(theta_deg: float) -> torch.Tensor:
    """H/max|H| — тот же масштаб, что даёт физически честная пара
    SLM(фаза)+Aperture(амплитуда): amplitude*exp(i*phase) =
    (|H|/max|H|)*(H/|H|) = H/max|H| — точно. Используется в
    IdealOpticalFrontend и в многоканальной физической системе."""
    tf = build_directional_transfer_function(theta_deg)
    return tf / tf.abs().max()


def _grid_spacing_meters() -> float:
    """Шаг расчётной сетки в метрах, как обычный float (без единиц pint) —
    нужен для torch.fft.fftfreq, который принимает только чистые числа."""
    spacing_quantity = GRID_SIZE / Nx
    try:
        return float(spacing_quantity.to(ureg.m).magnitude)
    except AttributeError:
        # на случай, если GRID_SIZE уже голое число без единиц
        return float(spacing_quantity) 


def build_directional_fft_kernel(theta_deg: float) -> torch.Tensor:
    """Передаточная функция для ПРЯМОГО FFT-фильтра (без линз/FreeSpace,
    без какого-либо физического распространения света).

    В отличие от build_directional_transfer_function (которая привязана
    к физической geometрии 4f-системы: масштаб kx задаётся линзой и
    фокусным расстоянием), здесь частоты определяются исключительно
    шагом сетки поля — через torch.fft.fftfreq, а не координату
    SLM-плоскости. Использовать build_directional_transfer_function
    внутри FFT-элемента было бы ошибкой: другой масштаб и другой
    порядок частот (fftfreq даёt 0,1,...,N/2-1,-N/2,...,-1, а не
    линейный центрированный порядок linspace).

    Нормировано на максимум амплитуды — тот же принцип, что и у
    build_ideal_normalized_transfer_function, чтобы все три варианта
    (физика / идеальный 4f / чистый FFT) были сравнимы по масштабу.
    """
    dx_spacing = _grid_spacing_meters()
    dy_spacing = dx_spacing  # сетка квадратная

    kx = 2 * torch.pi * torch.fft.fftfreq(Nx, d=dx_spacing)  # [Nx], в FFT-порядке
    ky = 2 * torch.pi * torch.fft.fftfreq(Ny, d=dy_spacing)  # [Ny], в FFT-порядке

    kx_grid = kx.unsqueeze(0).expand(Ny, Nx)
    ky_grid = ky.unsqueeze(1).expand(Ny, Nx)

    theta = math.radians(theta_deg)
    k_theta = kx_grid * math.cos(theta) + ky_grid * math.sin(theta)

    kernel = (1j * k_theta).to(torch.complex64)
    kernel = kernel / kernel.abs().max()
    return kernel.clone()


def derive_phase_amplitude_masks(transfer_function: torch.Tensor, eps: float = 1e-3):
    """Раскладывает произвольную комплексную передаточную функцию на
    физически реализуемые фазовую и амплитудную маски (SLM + Aperture).
    Amplitude нормирована на максимум (Aperture может только ослаблять).
    Phase отодвинута от границ [0, 2*pi] — иначе NaN при обучаемой маске
    (для фиксированной сейчас не критично, но оставлено для безопасности).
    """
    amplitude = transfer_function.abs()
    amplitude = amplitude / amplitude.max()

    phase = torch.angle(transfer_function) % (2 * torch.pi)
    phase = phase.clamp(eps, 2 * torch.pi - eps)

    return phase, amplitude


def build_x_derivative_phase_mask() -> torch.Tensor:
    """Фазовая часть d/dx — обратная совместимость со старым кодом."""
    tf = build_directional_transfer_function(CHANNEL_DIRECTIONS["dx"])
    phase, _ = derive_phase_amplitude_masks(tf)
    return phase


def build_x_derivative_amplitude_mask() -> torch.Tensor:
    """Амплитудная часть d/dx — обратная совместимость со старым кодом."""
    tf = build_directional_transfer_function(CHANNEL_DIRECTIONS["dx"])
    _, amplitude = derive_phase_amplitude_masks(tf)
    return amplitude
