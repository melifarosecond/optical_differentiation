import torch
from torch import nn
import torch.nn.functional as F
import svetlanna as sv
from svetlanna import Wavefront, LinearOpticalSetup

from optical_differentiation.svetlanna_data import (
    SIM_PARAMS, FOCAL_LENGTH, GRID_SIZE,
    build_x_derivative_phase_mask, build_x_derivative_amplitude_mask,
)


class SvetlannaOpticalFrontend(nn.Module):

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
            sv.elements.Aperture(
                SIM_PARAMS, mask=amplitude_mask,
            ),
            sv.elements.FreeSpace(SIM_PARAMS, distance=FOCAL_LENGTH, method="zpASM"),
            sv.elements.ThinLens(SIM_PARAMS, focal_length=FOCAL_LENGTH),
            sv.elements.FreeSpace(SIM_PARAMS, distance=FOCAL_LENGTH, method="zpASM"),
        ]

        self.setup = LinearOpticalSetup(elements)

    def forward(self, wavefront: Wavefront) -> Wavefront:
        return self.setup(wavefront)


class SvetlannaOpticalFrontend14(nn.Module):

    def __init__(self):
        super().__init__()
        self.core = SvetlannaOpticalFrontend()  

    def forward(self, wavefront: Wavefront) -> torch.Tensor:
        out = self.core(wavefront)
        intensity = out.intensity

        if intensity.dim() == 3:
            intensity = intensity.unsqueeze(1) 

        return F.adaptive_avg_pool2d(intensity, output_size=(14, 14))  


class SvetlannaOpticalModelEconomical(nn.Module):

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