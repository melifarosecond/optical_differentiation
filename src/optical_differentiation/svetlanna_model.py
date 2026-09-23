import torch
from torch import nn
import torch.nn.functional as F
import svetlanna as sv
from svetlanna import Wavefront, LinearOpticalSetup
from optical_differentiation.svetlanna_data import SIM_PARAMS, FOCAL_LENGTH, GRID_SIZE, CHANNEL_DIRECTIONS, OUTPUT_SIZE, build_x_derivative_phase_mask, build_x_derivative_amplitude_mask, build_ideal_normalized_transfer_function, build_directional_fft_kernel


def flip_4f_output(x):
    return torch.flip(x, dims=[-2, -1])


def crop_center_half(x):
    n = x.shape[-1]
    start = n // 4
    return x[..., start:start + n // 2, start:start + n // 2]


class SvetlannaOpticalFrontend(nn.Module):

    def __init__(self):
        super().__init__()
        phase_mask = build_x_derivative_phase_mask()
        amplitude_mask = build_x_derivative_amplitude_mask()
        elements = [sv.elements.FreeSpace(SIM_PARAMS, distance=FOCAL_LENGTH, method='zpASM'), sv.elements.ThinLens(SIM_PARAMS, focal_length=FOCAL_LENGTH), sv.elements.FreeSpace(SIM_PARAMS, distance=FOCAL_LENGTH, method='zpASM'), sv.elements.SpatialLightModulator(SIM_PARAMS, mask=phase_mask, height=GRID_SIZE, width=GRID_SIZE), sv.elements.Aperture(SIM_PARAMS, mask=amplitude_mask), sv.elements.FreeSpace(SIM_PARAMS, distance=FOCAL_LENGTH, method='zpASM'), sv.elements.ThinLens(SIM_PARAMS, focal_length=FOCAL_LENGTH), sv.elements.FreeSpace(SIM_PARAMS, distance=FOCAL_LENGTH, method='zpASM')]
        self.setup = LinearOpticalSetup(elements)

    def forward(self, wavefront):
        return self.setup(wavefront)


class IdealTransferFunctionElement(sv.elements.Element):

    def __init__(self, simulation_parameters, transfer_function):
        super().__init__(simulation_parameters=simulation_parameters)
        self.register_buffer('transfer_function', transfer_function)

    def forward(self, incident_wavefront):
        return Wavefront(incident_wavefront * self.transfer_function)


class IdealOpticalFrontend(nn.Module):

    def __init__(self):
        super().__init__()
        transfer_function = build_ideal_normalized_transfer_function(CHANNEL_DIRECTIONS['dx'])
        elements = [sv.elements.FreeSpace(SIM_PARAMS, distance=FOCAL_LENGTH, method='zpASM'), sv.elements.ThinLens(SIM_PARAMS, focal_length=FOCAL_LENGTH), sv.elements.FreeSpace(SIM_PARAMS, distance=FOCAL_LENGTH, method='zpASM'), IdealTransferFunctionElement(SIM_PARAMS, transfer_function), sv.elements.FreeSpace(SIM_PARAMS, distance=FOCAL_LENGTH, method='zpASM'), sv.elements.ThinLens(SIM_PARAMS, focal_length=FOCAL_LENGTH), sv.elements.FreeSpace(SIM_PARAMS, distance=FOCAL_LENGTH, method='zpASM')]
        self.setup = LinearOpticalSetup(elements)

    def forward(self, wavefront):
        return self.setup(wavefront)


class FFTDerivativeElement(sv.elements.Element):

    def __init__(self, simulation_parameters, kernel):
        super().__init__(simulation_parameters=simulation_parameters)
        self.register_buffer('kernel', kernel)

    def forward(self, incident_wavefront):
        field_fft = torch.fft.fft2(incident_wavefront)
        return Wavefront(torch.fft.ifft2(field_fft * self.kernel))


class FFTOpticalFrontend(nn.Module):

    def __init__(self):
        super().__init__()
        kernel = build_directional_fft_kernel(CHANNEL_DIRECTIONS['dx'])
        self.element = FFTDerivativeElement(SIM_PARAMS, kernel)

    def forward(self, wavefront):
        return self.element(wavefront)


class SvetlannaOpticalFrontendMultiChannel(nn.Module):

    def __init__(self, directions=('dx', 'dy', 'd45', 'd135')):
        super().__init__()
        self.channel_names = list(directions)
        self.front = LinearOpticalSetup([sv.elements.FreeSpace(SIM_PARAMS, distance=FOCAL_LENGTH, method='zpASM'), sv.elements.ThinLens(SIM_PARAMS, focal_length=FOCAL_LENGTH), sv.elements.FreeSpace(SIM_PARAMS, distance=FOCAL_LENGTH, method='zpASM')])
        self.back = LinearOpticalSetup([sv.elements.FreeSpace(SIM_PARAMS, distance=FOCAL_LENGTH, method='zpASM'), sv.elements.ThinLens(SIM_PARAMS, focal_length=FOCAL_LENGTH), sv.elements.FreeSpace(SIM_PARAMS, distance=FOCAL_LENGTH, method='zpASM')])
        transfer_functions = torch.stack([build_ideal_normalized_transfer_function(CHANNEL_DIRECTIONS[name]) for name in self.channel_names])
        self.register_buffer('transfer_functions', transfer_functions)

    def forward(self, wavefront):
        field_at_mask = self.front(wavefront)
        field_after = field_at_mask.unsqueeze(1) * self.transfer_functions.unsqueeze(0)
        B, C, Ny, Nx = field_after.shape
        field_flat = field_after.reshape(B * C, Ny, Nx)
        out = self.back(Wavefront(field_flat))
        return out.intensity.reshape(B, C, Ny, Nx)


class SvetlannaOpticalFrontendMultiChannel14(nn.Module):

    def __init__(self, directions=('dx', 'dy', 'd45', 'd135')):
        super().__init__()
        self.core = SvetlannaOpticalFrontendMultiChannel(directions)

    def forward(self, wavefront):
        intensity = self.core(wavefront)
        intensity = flip_4f_output(intensity)
        intensity = crop_center_half(intensity)
        return F.adaptive_avg_pool2d(intensity, output_size=(OUTPUT_SIZE, OUTPUT_SIZE))


class SvetlannaOpticalFrontend14(nn.Module):

    def __init__(self):
        super().__init__()
        self.core = SvetlannaOpticalFrontend()

    def forward(self, wavefront):
        intensity = self.core(wavefront).intensity
        if intensity.dim() == 3:
            intensity = intensity.unsqueeze(1)
        intensity = flip_4f_output(intensity)
        intensity = crop_center_half(intensity)
        return F.adaptive_avg_pool2d(intensity, output_size=(OUTPUT_SIZE, OUTPUT_SIZE))


class ElectronicHead(nn.Module):

    def __init__(self, n_channels=4, use_batchnorm=True):
        super().__init__()
        self.norm = nn.BatchNorm2d(n_channels) if use_batchnorm else nn.Identity()
        self.conv2 = nn.Conv2d(n_channels, 16, kernel_size=5, padding=2)
        self.pool = nn.AvgPool2d(2)
        self.fc1 = nn.Linear(16 * (OUTPUT_SIZE // 2) * (OUTPUT_SIZE // 2), 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.norm(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class SvetlannaOpticalModelEconomical(nn.Module):

    def __init__(self):
        super().__init__()
        self.frontend = SvetlannaOpticalFrontend14()
        self.head = ElectronicHead(n_channels=1)

    def forward(self, wavefront):
        return self.head(self.frontend(wavefront))


class SvetlannaOpticalModelMultiChannel(nn.Module):

    def __init__(self, directions=('dx', 'dy', 'd45', 'd135')):
        super().__init__()
        self.frontend = SvetlannaOpticalFrontendMultiChannel14(directions)
        self.head = ElectronicHead(n_channels=len(directions))

    def forward(self, wavefront):
        return self.head(self.frontend(wavefront))
