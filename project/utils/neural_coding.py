import math

import snntorch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from snntorch import spikegen
import torch
from torchvision.transforms.functional import affine


def neural_coding(images: torch.Tensor, neural_coding: str, timesteps: int) -> torch.Tensor:
    """Neural coding of the input image batch into a spiketrain

    Args:
        images (torch.Tensor): Batch of input images. tensor of dimension (B, C, H, W)

    Raises:
        NotImplementedError: If a certain neural coding is not implemented

    Returns:
        torch.Tensor: Resulting spiketrains of dimestion (T, B, C, H, W) where T is the total number of timesteps (i.e. timesteps)
    """
    if neural_coding == 'rate':
        return rate_coding(images, timesteps)
    elif neural_coding == 'ttfs':
        return ttfs(images, timesteps=timesteps, normalize=True, linear=False)
    elif neural_coding == 'phase':
        return phase_coding(images, timesteps=timesteps, is_weighted=True)
    elif neural_coding == 'burst':
        raise NotImplementedError('no')
        return burst_coding(images, burst_n_max, timesteps, burst_t_min)
    elif neural_coding == 'saccade':
        return saccade_coding(images, timesteps, 20, 20, 0.1)
    elif neural_coding == 'synchrony':
        return synchrony_coding(images, timesteps, 3, 0.1, 2)
    else:
        raise NotImplementedError()


def saccade_coding(images: torch.Tensor, timesteps: int = 100, max_dx: int = 20, max_dy: int = 20, delta_threshold=0.1):
    """Reproduces three saccades with delta modulation to obtain something similar to NMIST dataset.

    Args:
        images (torch.Tensor): batch of images with dimension (B, C, H, W)
        timesteps (int, optional): Number of timesteps of the final spike input. Defaults to 100.
        max_dx (int, optional): Max translation in x axis for the saccades. Defaults to 20.
        max_dy (int, optional): Max translation in y axis for the saccades. Defaults to 20.
        delta_threshold (float, optional): threshold used in delta modulation. Defaults to 0.1.

    Returns:
        torch.Tensor: result spike trains of dimension (T, B, C, H, W)
    """
    dx_step = max_dx / (2 * timesteps)  # pixel distance per timestep
    dy_step = max_dy / timesteps

    dx = 0.
    dy = 0.

    translations = torch.zeros((timesteps, *images.shape))
    i = 0
    # first saccade
    for _ in range(int(timesteps/3)):
        dx += dx_step
        dy += dy_step
        # print(dx, dy)
        translations[i] = affine(
            images, 0, [math.floor(dx), math.floor(dy)], 1, 0)
        i += 1

    # second saccade
    for _ in range(int(timesteps/3)):
        dx += dx_step
        dy = max(0, dy - dy_step)  # avoid negative value
        # print(dx, dy)
        translations[i] = affine(
            images, 0, [math.floor(dx), math.floor(dy)], 1, 0)
        i += 1

    # third saccade
    last_duration = timesteps - 2*int(timesteps/3)
    for _ in range(last_duration):
        dx = max(0, dx - 2 * dx_step)  # avoid negative value
        # print(dx, dy)
        translations[i] = affine(
            images, 0, [math.floor(dx), math.floor(dy)], 1, 0)
        i += 1

    return spikegen.delta(translations, threshold=delta_threshold)


def synchrony_coding(images: torch.Tensor, timesteps: int = 100, saccade_number: int = 3, delta_threshold: float = 0.1, dx: int = 2):
    """Synchrony-based coding (i.e. little saccades occuring during the whole timesteps.)

    Args:
        images (torch.Tensor): Batch of images with dimension (B, C, H, W)
        timesteps (int, optional): total timesteps for the resulting spiketrain. Defaults to 100.
        saccade_number (int, optional): Number of timesteps. Defaults to 3.
        delta_threshold (float, optional): threshold used for delta modulation. Defaults to 0.1.
        dx (int, optional): length (in pixels) for each translation used in the saccades. Defaults to 2.

    Returns:
        torch.Tensor: Resulting spiketrain of dimension (T, B, C, H, W)
    """
    translations = torch.zeros((timesteps, *images.shape))
    i = 0

    # compute time between 2 saccades
    rest_time = math.floor(timesteps / saccade_number)

    possible_translations = [
        [dx, dx],
        [dx, math.floor(dx/2)],
        [2 * dx, 0],
        [dx, 0],
        [0, 0]
    ]

    for j in range(saccade_number):
        translations[i] = affine(
            images, 0, possible_translations[j % len(possible_translations)], 1, 0)

        i += rest_time

    return spikegen.delta(translations, threshold=delta_threshold)


def burst_coding(images: torch.Tensor, N_max: int = 5, timesteps: int = 100, T_min: int = 2):
    # Compute N_s (the number of spikes per pixel)
    N_s = torch.ceil(N_max * images)

    # Compute ISI (the InterSpike Interval per pixel)
    ISI = torch.full_like(images, float(timesteps))

    ISI[N_s > 1.] = torch.ceil(-(timesteps - T_min)
                               * images[N_s > 1.] + timesteps)

    # Reconstruct the spikes tensor with N_s and ISI
    S = torch.zeros(
        (timesteps, images.shape[0], images.shape[1], images.shape[2], images.shape[3]))
    # first timesteps are full of 0s until T_min

    distances = torch.zeros_like(ISI)  # separating two spikes for each pixels
    for i in range(timesteps):
        mask = torch.logical_and(distances == ISI, N_s > 0)
        S[i] = mask.float()
        distances += 1
        distances[mask] = 0
        N_s[mask] -= 1

    return S


def rate_coding(images: torch.Tensor, timesteps: int = 100):
    return spikegen.rate(images, num_steps=timesteps)
    # return image_to_spikes(images, max_duration=timesteps, input_shape=images.shape[1:])


def phase_coding(images: torch.Tensor, timesteps: int = 100, is_weighted: bool = False):
    """Function that converts a grayscale image into spiketrains following the phase neural coding
    presented in https://www.frontiersin.org/articles/10.3389/fnins.2021.638474/full

    Args:
        image (np.ndarray): the grayscale image to convert of dimension (B, C, H, W)
        timesteps (int, optional): the number of timesteps. Must be a multiple of 8 (required for the phases). Defaults to 100.
        is_weighted (bool, optional): Flag that indicates whether the output spikes are weighted following the w_s parameter defined in the original paper. Defaults to True.

    Returns:
        np.ndarray: the spike tensor of dimension (T, H, W)
    """
    # compute number of periods
    periods = (timesteps // 8) + 1

    # convert to numpy because we have to
    images = (images * 255).cpu().numpy().astype(np.uint8)

    # binary representation of the image (it makes 8 )
    bit_representation = np.unpackbits(
        images[..., None], axis=-1).transpose(-1, 0, 1, 2, 3).astype(np.float32)

    # IF the weighted input option is used
    if is_weighted:
        # obtained with the equation seen in the referenced paper
        w_s = [0.5, 0.25, 0.125, 0.0625, 0.0313,
               0.015625, 0.0078125, 0.00390625]
        for i, weight in enumerate(w_s):
            bit_representation[i, :, :, :,
                               :] = bit_representation[i, :, :, :, :] * weight

    # Repeat the bit representation to create the final output spikes
    S = np.tile(bit_representation, (periods, 1, 1, 1, 1))[
        0:timesteps, :, :, :, :]

    return torch.from_numpy(S)


def P_th(t: float, theta_0: float, tau_th: float):
    # calculate Pth(t)
    P_th = theta_0 * math.exp(-t/tau_th)
    return P_th


def ttfs(images: np.ndarray, timesteps: int, normalize: bool, linear: bool):
    return spikegen.latency(images, num_steps=timesteps, normalize=normalize, linear=linear)


if __name__ == '__main__':
    theta_0 = 1.0
    tau_th = 6.0
    ts = 80

    image = cv2.imread('input.jpg', cv2.IMREAD_GRAYSCALE)
    # S = ttfs(image, theta_0, tau_th, ts)
    S = phase_coding(image, ts, is_weighted=False)
    print(S.shape)
    exit()
    spikes = (S == 1.).sum()
    nonspikes = (S == 0.).sum()

    print(spikes, nonspikes)

    print('Spikes !!!!!')
    for t in range(ts):
        print((S[t] == 1.).sum())
