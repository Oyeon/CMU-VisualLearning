import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from utils import (
    cosine_beta_schedule,
    default,
    extract,
    unnormalize_to_zero_to_one,
)
from einops import rearrange, reduce

class DiffusionModel(nn.Module):
    def __init__(
        self,
        model,
        timesteps=1000,
        sampling_timesteps=None,
        ddim_sampling_eta=1.,
    ):
        super(DiffusionModel, self).__init__()

        self.model = model
        self.channels = self.model.channels
        self.device = torch.cuda.current_device()

        self.betas = cosine_beta_schedule(timesteps).to(self.device)
        self.num_timesteps = self.betas.shape[0]

        alphas = 1. - self.betas

        # Compute cumulative products for current and previous timesteps
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # Pre-compute values needed for forward process
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        # Coefficients for predicting x_0
        self.x_0_pred_coef_1 = 1.0 / self.sqrt_alphas_cumprod
        self.x_0_pred_coef_2 = -self.sqrt_one_minus_alphas_cumprod / self.sqrt_alphas_cumprod

        # Compute the coefficients for the mean
        self.sqrt_alphas = torch.sqrt(alphas)
        self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev)) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = ((1. - self.alphas_cumprod_prev) * self.sqrt_alphas) / (1. - self.alphas_cumprod)

        # Compute posterior variance
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

        self.posterior_log_variance_clipped = torch.log(
            self.posterior_variance.clamp(min=1e-20))

        # Sampling related parameters
        self.sampling_timesteps = default(sampling_timesteps, timesteps)  # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

    def get_posterior_parameters(self, x_0, x_t, t):
        # Compute the posterior mean and variance for x_{t-1}
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_0 +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x_t, t):
        # Predict the noise using the denoising model
        pred_noise = self.model(x_t, t)

        # Compute x_0 using the pre-computed coefficients
        coef1 = extract(self.x_0_pred_coef_1, t, x_t.shape)
        coef2 = extract(self.x_0_pred_coef_2, t, x_t.shape)
        x_0 = coef1 * x_t + coef2 * pred_noise

        # Clamp x_0 between -1 and 1
        x_0 = torch.clamp(x_0, -1., 1.)

        return pred_noise, x_0

    @torch.no_grad()
    def predict_denoised_at_prev_timestep(self, x, t: int):
        # Get model predictions
        pred_noise, x_0 = self.model_predictions(x, t)

        # Compute posterior parameters
        posterior_mean, posterior_variance, posterior_log_variance_clipped = self.get_posterior_parameters(x_0, x, t)

        # Sample from the posterior distribution
        noise = torch.randn_like(x) if t[0] > 0 else torch.zeros_like(x)
        pred_img = posterior_mean + torch.sqrt(posterior_variance) * noise

        return pred_img, x_0

    @torch.no_grad()
    def sample_ddpm(self, shape, z):
        img = z
        for t in tqdm(range(self.num_timesteps - 1, -1, -1)):
            batched_times = torch.full((img.shape[0],), t, device=self.device, dtype=torch.long)
            img, _ = self.predict_denoised_at_prev_timestep(img, batched_times)
        img = unnormalize_to_zero_to_one(img)
        return img

    def sample_times(self, total_timesteps, sampling_timesteps):
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        return list(reversed(times.int().tolist()))

    def get_time_pairs(self, times):
        return list(zip(times[:-1], times[1:]))

    def ddim_step(self, batch, device, tau_i, tau_isub1, img, model_predictions, alphas_cumprod, eta):
        # Ensure tau_isub1 is at least 0
        tau_isub1 = max(tau_isub1, 0)

        # Create timestep tensors
        t = torch.full((batch,), tau_i, device=device, dtype=torch.long)
        t_prev = torch.full((batch,), tau_isub1, device=device, dtype=torch.long)

        # Predict noise and x_0
        pred_noise, x_0 = model_predictions(img, t)

        # Compute alphas for current and previous timesteps
        alpha_t = extract(alphas_cumprod, t, img.shape)
        alpha_t_prev = extract(alphas_cumprod, t_prev, img.shape)

        # Compute sigma for the current timestep
        sigma_t = eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev))

        # Compute the direction pointing to x_t
        pred_dir = torch.sqrt(1 - alpha_t_prev - sigma_t ** 2) * pred_noise

        # Compute the predicted image at the previous timestep
        img = torch.sqrt(alpha_t_prev) * x_0 + pred_dir + sigma_t * torch.randn_like(img)

        return img, x_0

    def sample_ddim(self, shape, z):
        batch, device, total_timesteps, sampling_timesteps, eta = shape[0], self.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta

        times = self.sample_times(total_timesteps, sampling_timesteps)
        time_pairs = self.get_time_pairs(times)

        img = z
        for tau_i, tau_isub1 in tqdm(time_pairs, desc='sampling loop time step'):
            img, _ = self.ddim_step(batch, device, tau_i, tau_isub1, img, self.model_predictions, self.alphas_cumprod, eta)

        img = unnormalize_to_zero_to_one(img)
        return img

    @torch.no_grad()
    def sample(self, shape):
        sample_fn = self.sample_ddpm if not self.is_ddim_sampling else self.sample_ddim
        z = torch.randn(shape, device=self.betas.device)
        return sample_fn(shape, z)

    @torch.no_grad()
    def sample_given_z(self, z, shape):
        sample_fn = self.sample_ddpm if not self.is_ddim_sampling else self.sample_ddim
        z = z.reshape(shape)
        return sample_fn(shape, z)
