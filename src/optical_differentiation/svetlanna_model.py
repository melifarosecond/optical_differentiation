import torch
from torch import nn
import torch.nn.functional as F
import svetlanna as sv
from svetlanna import Wavefront, LinearOpticalSetup

from optical_differentiation.svetlanna_data import (
    SIM_PARAMS, FOCAL_LENGTH, GRID_SIZE, CHANNEL_DIRECTIONS,
    build_x_derivative_phase_mask, build_x_derivative_amplitude_mask,
    build_directional_transfer_function, build_ideal_normalized_transfer_function,
    build_directional_fft_kernel,
)


# =====================================================================
# Уровень 1: честная физика — SLM + Aperture, фиксированы
# =====================================================================

class SvetlannaOpticalFrontend(nn.Module):
    """Физический прообраз метаповерхности: 4f-система с ФИКСИРОВАННЫМИ
    (необучаемыми) SLM (фаза) + Aperture (амплитуда) на одной оптической
    плоскости (без FreeSpace между ними) — вместе реализуют d/dx.
    """

    def __init__(self):
        super().__init__()

        phase_mask = build_x_derivative_phase_mask()
        amplitude_mask = build_x_derivative_amplitude_mask()

        elements = [
            sv.elements.FreeSpace(SIM_PARAMS, distance=FOCAL_LENGTH, method="zpASM"),
            sv.elements.ThinLens(SIM_PARAMS, focal_length=FOCAL_LENGTH),
            sv.elements.FreeSpace(SIM_PARAMS, distance=FOCAL_LENGTH, method="zpASM"),

            sv.elements.SpatialLightModulator(
                SIM_PARAMS, mask=phase_mask, height=GRID_SIZE, width=GRID_SIZE,
            ),
            sv.elements.Aperture(SIM_PARAMS, mask=amplitude_mask),

            sv.elements.FreeSpace(SIM_PARAMS, distance=FOCAL_LENGTH, method="zpASM"),
            sv.elements.ThinLens(SIM_PARAMS, focal_length=FOCAL_LENGTH),
            sv.elements.FreeSpace(SIM_PARAMS, distance=FOCAL_LENGTH, method="zpASM"),
        ]

        self.setup = LinearOpticalSetup(elements)

    def forward(self, wavefront: Wavefront) -> Wavefront:
        return self.setup(wavefront)


class SvetlannaOpticalFrontend14(nn.Module):
    """SvetlannaOpticalFrontend + сжатие 200x200 -> 14x14 усреднением."""

    def __init__(self):
        super().__init__()
        self.core = SvetlannaOpticalFrontend()

    def forward(self, wavefront: Wavefront) -> torch.Tensor:
        out = self.core(wavefront)
        intensity = out.intensity
        if intensity.dim() == 3:
            intensity = intensity.unsqueeze(1)
        return F.adaptive_avg_pool2d(intensity, output_size=(14, 14))  # [B, 1, 14, 14]


class SvetlannaOpticalModelEconomical(nn.Module):
    """Полная одноканальная модель: физика (d/dx) + электроника без conv1."""

    def __init__(self):
        super().__init__()
        self.frontend = SvetlannaOpticalFrontend14()
        self.conv2 = nn.Conv2d(1, 16, kernel_size=5, padding=2)
        self.pool = nn.AvgPool2d(2)
        self.fc1 = nn.Linear(16 * 7 * 7, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, wavefront: Wavefront) -> torch.Tensor:
        x = self.frontend(wavefront)
        x = self.pool(F.relu(self.conv2(x)))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# =====================================================================
# Уровень 2: "идеальный" эталон — один custom element вместо SLM+Aperture,
# НО всё ещё внутри физического каркаса (линзы + FreeSpace)
# =====================================================================

class IdealTransferFunctionElement(sv.elements.Element):
    """Элемент, напрямую умножающий поле на заданную комплексную
    передаточную функцию — без разбиения на физически отдельные
    фазу и амплитуду. Используется как эталон для сравнения точности
    с физически разложенной SLM+Aperture реализацией."""

    def __init__(self, simulation_parameters, transfer_function: torch.Tensor):
        super().__init__(simulation_parameters=simulation_parameters)
        self.register_buffer("transfer_function", transfer_function)

    def forward(self, incident_wavefront: Wavefront) -> Wavefront:
        return Wavefront(incident_wavefront * self.transfer_function)


class IdealOpticalFrontend(nn.Module):
    """"Идеальная" одноканальная (d/dx) 4f-система: тот же ThinLens/FreeSpace
    каркас, что у SvetlannaOpticalFrontend, но с IdealTransferFunctionElement
    вместо пары SLM+Aperture. Нормировка (H/max|H|) намеренно совпадает
    с эффективной передаточной функцией физической версии — сравнение
    честное, разница объясняется только способом реализации, не масштабом.

    В отличие от FFTOpticalFrontend (ниже) — здесь поле физически проходит
    через линзы и FreeSpace, преобразование Фурье возникает из дифракции,
    а не считается напрямую числовым FFT.
    """

    def __init__(self):
        super().__init__()
        transfer_function = build_ideal_normalized_transfer_function(CHANNEL_DIRECTIONS["dx"])

        elements = [
            sv.elements.FreeSpace(SIM_PARAMS, distance=FOCAL_LENGTH, method="zpASM"),
            sv.elements.ThinLens(SIM_PARAMS, focal_length=FOCAL_LENGTH),
            sv.elements.FreeSpace(SIM_PARAMS, distance=FOCAL_LENGTH, method="zpASM"),

            IdealTransferFunctionElement(SIM_PARAMS, transfer_function),

            sv.elements.FreeSpace(SIM_PARAMS, distance=FOCAL_LENGTH, method="zpASM"),
            sv.elements.ThinLens(SIM_PARAMS, focal_length=FOCAL_LENGTH),
            sv.elements.FreeSpace(SIM_PARAMS, distance=FOCAL_LENGTH, method="zpASM"),
        ]

        self.setup = LinearOpticalSetup(elements)

    def forward(self, wavefront: Wavefront) -> Wavefront:
        return self.setup(wavefront)


# =====================================================================
# Уровень 3: чистый FFT-фильтр — без линз, без FreeSpace, без физического
# распространения света вообще. Не модель метаповерхности — быстрый
# числовой эталон для оценки предела точности.
# =====================================================================

class FFTDerivativeElement(sv.elements.Element):
    """Прямой числовой фильтр: FFT -> умножение на ядро -> IFFT,
    БЕЗ физического распространения света. Использует
    build_directional_fft_kernel (частоты по шагу сетки, torch.fft.fftfreq),
    а НЕ build_directional_transfer_function (та привязана к геометрии
    физической 4f-системы и здесь была бы неверна по масштабу и порядку
    частот)."""

    def __init__(self, simulation_parameters, kernel: torch.Tensor):
        super().__init__(simulation_parameters=simulation_parameters)
        self.register_buffer("kernel", kernel)

    def forward(self, incident_wavefront: Wavefront) -> Wavefront:
        field_fft = torch.fft.fft2(incident_wavefront)
        filtered = field_fft * self.kernel
        result = torch.fft.ifft2(filtered)
        return Wavefront(result)


class FFTOpticalFrontend(nn.Module):
    """Одноканальный (d/dx) "фронтенд" на чистом FFT — без линз и
    FreeSpace вообще. Самый дешёвый и самый быстрый из трёх вариантов,
    но физически не привязан ни к какому оптическому устройству.
    Отвечает на вопрос "какой предел точности в принципе достижим",
    а не "как это сделать в реальном устройстве"."""

    def __init__(self):
        super().__init__()
        kernel = build_directional_fft_kernel(CHANNEL_DIRECTIONS["dx"])
        self.element = FFTDerivativeElement(SIM_PARAMS, kernel)

    def forward(self, wavefront: Wavefront) -> Wavefront:
        return self.element(wavefront)


# =====================================================================
# 4 физических канала (dx, dy, d45, d135)
# =====================================================================

class SvetlannaOpticalFrontendMultiChannel(nn.Module):
    """4 независимых физических канала. Общие FreeSpace/ThinLens части
    (до и после маски) считаются один раз, branching (разные передаточные
    функции по каналам) сделан батчингом, а не тайлингом пространства —
    раскидать каналы по четвертям входной/частотной плоскости физически
    некорректно (сдвинутые копии дают перекрывающиеся, а не разделённые
    спектры — теорема сдвига Фурье).

    Каждый канал математически эквивалентен отдельной физической
    SLM+Aperture паре (см. IdealOpticalFrontend/compare_three_levels.py
    для проверки точности этой эквивалентности).
    """

    def __init__(self, directions=("dx", "dy", "d45", "d135")):
        super().__init__()
        self.channel_names = list(directions)

        self.front = LinearOpticalSetup([
            sv.elements.FreeSpace(SIM_PARAMS, distance=FOCAL_LENGTH, method="zpASM"),
            sv.elements.ThinLens(SIM_PARAMS, focal_length=FOCAL_LENGTH),
            sv.elements.FreeSpace(SIM_PARAMS, distance=FOCAL_LENGTH, method="zpASM"),
        ])
        self.back = LinearOpticalSetup([
            sv.elements.FreeSpace(SIM_PARAMS, distance=FOCAL_LENGTH, method="zpASM"),
            sv.elements.ThinLens(SIM_PARAMS, focal_length=FOCAL_LENGTH),
            sv.elements.FreeSpace(SIM_PARAMS, distance=FOCAL_LENGTH, method="zpASM"),
        ])

        transfer_functions = torch.stack([
            build_ideal_normalized_transfer_function(CHANNEL_DIRECTIONS[name])
            for name in self.channel_names
        ])  # [C, Ny, Nx], комплексные, каждый канал нормирован на свой максимум
        self.register_buffer("transfer_functions", transfer_functions)

    def forward(self, wavefront: Wavefront) -> torch.Tensor:
        field_at_mask = self.front(wavefront)  # [B, Ny, Nx]

        # branching: [B, Ny, Nx] x [C, Ny, Nx] -> [B, C, Ny, Nx]
        field_after = field_at_mask.unsqueeze(1) * self.transfer_functions.unsqueeze(0)

        B, C, Ny, Nx = field_after.shape
        field_flat = field_after.reshape(B * C, Ny, Nx)

        out = self.back(Wavefront(field_flat))
        intensity = out.intensity.reshape(B, C, Ny, Nx)
        return intensity  # [B, C, Ny, Nx]


class SvetlannaOpticalFrontendMultiChannel14(nn.Module):
    """SvetlannaOpticalFrontendMultiChannel + сжатие каждого канала
    200x200 -> 14x14 усреднением."""

    def __init__(self, directions=("dx", "dy", "d45", "d135")):
        super().__init__()
        self.core = SvetlannaOpticalFrontendMultiChannel(directions)

    def forward(self, wavefront: Wavefront) -> torch.Tensor:
        intensity = self.core(wavefront)  # [B, C, 200, 200]
        return F.adaptive_avg_pool2d(intensity, output_size=(14, 14))  # [B, C, 14, 14]


class SvetlannaOpticalModelMultiChannel(nn.Module):
    """Полная 4-канальная модель — прямой аналог OpticalModelDiagonal
    из PyTorch-части, но с честной физической дифракцией вместо
    программных сверточных ядер dx/dy/45°/135°."""

    def __init__(self, directions=("dx", "dy", "d45", "d135")):
        super().__init__()
        n_channels = len(directions)
        self.frontend = SvetlannaOpticalFrontendMultiChannel14(directions)
        self.conv2 = nn.Conv2d(n_channels, 16, kernel_size=5, padding=2)
        self.pool = nn.AvgPool2d(2)
        self.fc1 = nn.Linear(16 * 7 * 7, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, wavefront: Wavefront) -> torch.Tensor:
        x = self.frontend(wavefront)              # [B, 4, 14, 14]
        x = self.pool(F.relu(self.conv2(x)))       # [B, 16, 7, 7]
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
