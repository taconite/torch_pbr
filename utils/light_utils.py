import torch
import torch.nn.functional as F
import numpy as np


def compute_energy(lgtSGs):
    """Compute the total energy of a light source represented as mixture of
    Spherical Gaussians (SGs).  The energy is computed as the integral of the
    SGs over an unit sphere.
    reference: https://github.com/Kai-46/PhySG/blob/master/code/model/sg_envmap_material.py
    Args:
        lgtSGs: (N, 7) tensor of SG parameters, where N is the number of SGs.
    Returns:
        energy: (N, 1) tensor of energy
    """
    lgtLambda = torch.abs(lgtSGs[:, 3:4])
    lgtMu = torch.abs(lgtSGs[:, 4:])
    energy = lgtMu * 2.0 * np.pi / lgtLambda * (1.0 - torch.exp(-2.0 * lgtLambda))
    return energy


def fibonacci_sphere(samples=1):
    '''Uniformly distribute points on a sphere
    Args:
        samples : number of points
    Returns:
        points : (samples, 3) numpy array
    reference: https://github.com/Kai-46/PhySG/blob/master/code/model/sg_envmap_material.py
    '''
    points = []
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians
    for i in range(samples):
        z = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - z * z)  # radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius
        y = np.sin(theta) * radius

        points.append([x, y, z])
    points = np.array(points)
    return points


def eval_SGs(lgtSGs, viewdirs):
    """Evaluate the light source represented as mixture of Spherical Gaussians
    (SGs) under the given view directions.
    Args:
        lgtSGs: (N, 7) tensor of SG parameters, where N is the number of SGs.
        viewdirs: (..., 3) tensor of view directions.
    Returns:
        Lo: (..., 3) tensor of radiance values.
    """
    viewdirs = viewdirs
    viewdirs = viewdirs[..., None, :]  # [..., 1, 3]

    # [N, 7] ---> [..., N, 7]
    lgtSGs = lgtSGs.expand(list(viewdirs.shape[:-2]) + list(lgtSGs.shape))

    lgtSGLobes = F.normalize(lgtSGs[..., :3], dim=-1)  # [..., N, 3]
    lgtSGLambdas = torch.abs(lgtSGs[..., 3:4])  # [..., N, 1]
    lgtSGMus = torch.abs(lgtSGs[..., -3:])  # [..., N, 3]
    # [..., N, 3]
    Lo = lgtSGMus * torch.exp(
        lgtSGLambdas * (torch.sum(viewdirs * lgtSGLobes, dim=-1, keepdim=True) - 1.0)
    )
    Lo = torch.sum(Lo, dim=-2)  # [..., 3]
    return Lo
