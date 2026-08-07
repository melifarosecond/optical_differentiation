import torch
from torch import nn
import svetlanna as sv
from svetlanna import Wavefront, LinearOpticalSetup

from optical_differentiation.svetlanna_data import (
    SIM_PARAMS, FOCAL_LENGTH, GRID_SIZE, Nx, Ny,
    build_detector_masks, build_x_derivative_phase_mask,
)


class OpticalSystem4F(nn.Module):

    def __init__(self, slm_init_mask: torch.Tensor | None = None):
        super().__init__()

        if slm_init_mask is None:
            slm_init_mask = build_x_derivative_phase_mask()

        elements = [
            sv.elements.FreeSpace(SIM_PARAMS, distance=FOCAL_LENGTH, method="ASM"),
            sv.elements.ThinLens(SIM_PARAMS, focal_length=FOCAL_LENGTH),
            sv.elements.FreeSpace(SIM_PARAMS, distance=FOCAL_LENGTH, method="ASM"),
            sv.elements.SpatialLightModulator(
                SIM_PARAMS,
                mask=sv.ConstrainedParameter(slm_init_mask, min_value=0, max_value=2 * torch.pi),
                height=GRID_SIZE,  
                width=GRID_SIZE,
            ),
            sv.elements.FreeSpace(SIM_PARAMS, distance=FOCAL_LENGTH, method="ASM"),
            sv.elements.ThinLens(SIM_PARAMS, focal_length=FOCAL_LENGTH),
            sv.elements.FreeSpace(SIM_PARAMS, distance=FOCAL_LENGTH, method="ASM"),
        ]

        self.setup = LinearOpticalSetup(elements)

    def forward(self, wavefront: Wavefront) -> Wavefront:
        return self.setup(wavefront)


class OpticalClassifier(nn.Module):
    
    def __init__(self, slm_init_mask: torch.Tensor | None = None):
        super().__init__()
        self.optical_core = OpticalSystem4F(slm_init_mask)
        self.register_buffer("detector_masks", build_detector_masks())

    def forward(self, wavefront: Wavefront):
        out_wavefront = self.optical_core(wavefront)
        intensity = out_wavefront.intensity

        I_l = (intensity[..., None] * self.detector_masks).sum(dim=(-2, -3))
        I_l_norm = I_l / (torch.max(I_l, dim=-1, keepdim=True).values + 1e-8) * 10
        return I_l_norm