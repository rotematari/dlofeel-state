import torch
import torch.nn as nn

try:
    import torchcurves.functional as tcf  # Assumed import for torchcurves
except ImportError:
    print("Warning: torchcurves not installed. Spline layer will fail.")
    tcf = None


class DifferentiableBSpline(nn.Module):
    def __init__(self, n_control_points, n_output_points, degree=3):
        super().__init__()
        self.n_control_points = n_control_points
        self.n_output_points = n_output_points
        self.degree = degree

        # Register sample points (t) buffer
        # Domain [-1, 1] matches torchcurves example for uniform knots
        t = torch.linspace(-1, 1, n_output_points)
        self.register_buffer('t', t)

    def forward(self, control_points,knots=None):
        """
        Args:
            control_points: [B, n_control_points, 3]
            knots: [B, n_control_points + degree + 1]
        Returns:
            smoothed_points: [B, n_output_points, 3]
        """
        if tcf is None:
            raise ImportError("torchcurves is not installed")

        B = control_points.shape[0]
        # spline_args: [n_output, B] (parameter values t)

        # NOTE: torchcurves backward pass 'scatter_add_' fails with AMP if dtypes mismatch.
        # We force float32 execution for the spline op to ensure stability and compatibility.
        control_points_fp32 = control_points.float()
        spline_args_fp32 = self.t.unsqueeze(1).expand(-1, B).float().contiguous()
        if knots is not None:
            knots_fp32 = knots.float().contiguous()
        else:
            knots_fp32 = None

        # Output shape from doc: [n_output, B, Dim]
        curve_points = tcf.bspline_curves(spline_args_fp32, control_points_fp32, knots=knots_fp32, degree=self.degree)

        # Reshape to [B, n_output, Dim] and cast back to original dtype
        return curve_points.permute(1, 0, 2).to(control_points.dtype)
