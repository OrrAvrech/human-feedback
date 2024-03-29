import random
from scipy.spatial.transform import Rotation
from smplx.lbs import batch_rigid_transform
import numpy as np
import torch


def generate_random_rotation_matrices(batch_size, num_joints, num_indices=1):
    # Initialize with identity rotation matrices
    rotations = np.tile(np.eye(3)[None, None, :, :], (batch_size, num_joints, 1, 1))

    # Randomly sample indices for each batch
    joint_arr = np.arange(10, num_joints)
    sampled_indices = np.random.choice(joint_arr, size=(batch_size, num_indices))

    # Generate random rotation matrices for the sampled indices
    sampled_rotations = Rotation.random(batch_size * num_indices).as_matrix()
    rotations[np.arange(batch_size)[:, None], sampled_indices, :, :] = sampled_rotations.reshape(batch_size, num_indices, 3, 3)

    # Convert to PyTorch tensor
    rotations_tensor = torch.Tensor(rotations)

    return rotations_tensor


def batch_linear_interpolation(point1, point2, window_size):
    alphas = np.zeros((window_size, *point1.shape))
    for d in range(3):
        alphas[..., d] = np.linspace(point1[:, d], point2[:, d], window_size)
    return alphas


def random_motion_warping(joints, parents, pert_perc, num_indices=1):
    num_frames = joints.shape[0]
    num_joints = joints.shape[1]
    rotation_mats = generate_random_rotation_matrices(num_frames, num_joints, num_indices)
    pert_frames = max(int(num_frames * pert_perc), 2)
    jump = num_frames // pert_frames
    window_size = int(jump // random.uniform(1.5, 3))
    pert_indices = sample_integers_with_spacing(pert_frames, 0, num_frames, window_size)

    pert_joints = joints[pert_indices, ...]
    pert_rot_mats = rotation_mats[pert_indices, ...]

    posed_joints, _ = batch_rigid_transform(pert_rot_mats, pert_joints, parents)
    posed_joints = posed_joints.detach().cpu().numpy()

    warped_joints = joints.detach().cpu().numpy()
    pert_frame_indices = []
    for posed_idx, i in enumerate(pert_indices):
        pert_frame_indices += list(range(i - window_size, i + window_size + 1))

        point_before_1 = warped_joints[i - window_size, ...]
        point_before_2 = posed_joints[posed_idx, ...]
        vec_before = batch_linear_interpolation(point_before_1, point_before_2, window_size)

        point_after_1 = posed_joints[posed_idx, ...]
        point_after_2 = warped_joints[i + window_size, ...]
        vec_after = batch_linear_interpolation(point_after_1, point_after_2, window_size)

        warped_joints[i - window_size : i] = vec_before
        warped_joints[i : i + window_size] = vec_after

    warped_joints = torch.Tensor(warped_joints)
    return warped_joints, pert_frame_indices


def sample_integers_with_spacing(x, start, end, window_size):
    """
    Uniformly sample x numbers from the range [start, end], ensuring they are at least window_size apart.

    Parameters:
    - x: Number of samples to generate.
    - start: Start of the range.
    - end: End of the range.
    - window_size: Minimum spacing between samples.

    Returns:
    - A numpy array of x uniformly spaced numbers within the specified range.
    """
    # Ensure there is enough room for x samples with minimum spacing
    if x * window_size > (end - start):
        raise ValueError("Cannot generate x samples with the specified window size within the given range.")

    # Generate x random positions with minimum spacing
    positions = np.sort(np.random.randint(start + window_size, end - x * window_size, x))

    # Adjust positions to ensure minimum spacing
    positions += np.arange(x) * window_size
    positions[-1] = min(positions[-1], end-1)

    return positions
