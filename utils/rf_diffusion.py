import torch
from torchdiffeq import odeint
import time


# ---------------------------------------------------------
# Compute mean over all non-batch dimensions
# ---------------------------------------------------------
def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    Used for computing scalar loss per sample.
    """
    return tensor.flatten(1).mean(1)

    
# ---------------------------------------------------------
# Cosine-based time mapping (from RF-Diffusion paper)
# Algorithm 21 in https://arxiv.org/abs/2403.03206
# ---------------------------------------------------------
def cosmap(t):
    """
    Map uniform time t in [0,1] to a cosine-like schedule.
    This controls the noise interpolation.
    """
    return 1. - (1. / (torch.tan(torch.pi / 2 * t) + 1))


# ---------------------------------------------------------
# Append singleton dimensions for broadcasting
# ---------------------------------------------------------
def append_dims(t, ndims):
    """
    Expand tensor by appending ndims singleton dimensions.
    Used to match tensor shapes for broadcasting.
    """
    shape = t.shape
    return t.reshape(*shape, *((1,) * ndims))


# ---------------------------------------------------------
# Compute RF-Diffusion training loss
# ---------------------------------------------------------
def rf_training_losses(
    model,
    x_start,
    micro_text,
    f0_channels=32,
    predict="flow"
):
    """
    Compute RF-Diffusion training loss for one batch.

    Args:
        model: diffusion model
        x_start: clean latent (B, N, C)
        micro_text: text embedding / prompt
        f0_channels: number of static channels
        predict: "flow" or "noise"

    Returns:
        terms: dictionary containing training losses
    """

    # Sample Gaussian noise
    noise = torch.randn_like(x_start)

    # Sample random timesteps in [0,1]
    times = torch.rand(x_start.shape[0], device=x_start.device)

    # Expand time dimensions for broadcasting
    padded_times = append_dims(times, x_start.ndim - 1)

    # Apply cosine mapping
    t = cosmap(padded_times)

    # Interpolate between clean data and noise
    x_t = t * x_start + (1. - t) * noise

    # Keep static part (f0) unchanged
    x_t = torch.cat(
        [x_start[:, :, :f0_channels],
         x_t[:, :, f0_channels:]],
        dim=-1
    )

    # Ground-truth flow
    flow = x_start - noise

    terms = {}

    # Model prediction
    model_output = model(
        x_t,
        t.squeeze(-1).squeeze(-1).squeeze(-1),
        micro_text
    )

    # Select training target
    if predict == "flow":
        target = flow
    elif predict == "noise":
        target = noise
    else:
        raise ValueError(f"Unknown objective {predict}")

    # Number of deformable channels
    ft_channels = x_start.shape[-1] - f0_channels

    # MSE loss on deformation channels
    terms["loss"] = mean_flat(
        (target[:, :, -ft_channels:]
         - model_output[:, :, -ft_channels:]) ** 2
    )

    return terms


# ---------------------------------------------------------
# RF-Diffusion sampling via ODE solver
# ---------------------------------------------------------
@torch.no_grad()
def rf_sample(
    model,
    shape,
    steps=64,
    text_prmpt=[""],
    device=None,
    guidance_scale=3.0,
    predict="flow",
    f0=None,
):
    """
    Sample from RF-Diffusion model using ODE integration.

    Args:
        model: trained diffusion model
        shape: latent shape (B, N, C)
        steps: number of ODE steps
        text_prmpt: text prompts (for classifier-free guidance)
        device: torch device
        guidance_scale: CFG scale
        predict: "flow" or "noise"
        f0: static latent part

    Returns:
        sampled_data: generated latent
    """

    # Number of static channels
    f0_channels = f0.shape[-1]

    # ODE solver settings
    odeint_kwargs = dict(
        atol=1e-5,
        rtol=1e-5,
        method="midpoint"
    )


    # -----------------------------------------------------
    # ODE dynamics function
    # -----------------------------------------------------
    def ode_fn(t, x):
        """
        Compute dx/dt at time t.
        Used by ODE solver.
        """

        # Duplicate batch for CFG (cond + uncond)
        x = torch.cat([x] * 2)

        # Inject static f0 channels
        x = torch.cat(
            [
                f0.repeat(x.shape[0], 1, 1),
                x[:, :, f0_channels:]
            ],
            dim=-1
        )

        # Predict flow / noise
        if predict == "flow":
            flow = model(
                x,
                t.unsqueeze(0).repeat(x.shape[0]),
                text_prmpt
            )

        elif predict == "noise":

            noise = model(
                x,
                t.unsqueeze(0).repeat(x.shape[0]),
                text_prmpt
            )

            padded_times = append_dims(t, noise.ndim - 1)

            # Convert noise prediction to flow
            flow = (x - noise) / padded_times.clamp(min=1e-10)

        else:
            raise ValueError(f"Unknown objective {predict}")

        # Split conditional and unconditional outputs
        cond_flow, uncond_flow = torch.split(
            flow,
            len(flow) // 2,
            dim=0
        )

        # Classifier-Free Guidance
        flow = uncond_flow + guidance_scale * (
            cond_flow - uncond_flow
        )

        return flow


    # -----------------------------------------------------
    # Initialize with Gaussian noise
    # -----------------------------------------------------
    noise = torch.randn(*shape, device=device)

    # Integration time grid
    times = torch.linspace(0., 1., steps, device=device)


    # -----------------------------------------------------
    # Solve ODE
    # -----------------------------------------------------
    trajectory = odeint(
        ode_fn,
        noise,
        times,
        **odeint_kwargs
    )

    # Take final step
    sampled_data = trajectory[-1][:1]

    # Restore static part
    sampled_data = torch.cat(
        [f0, sampled_data[:, :, f0_channels:]],
        dim=-1
    )

    return sampled_data
