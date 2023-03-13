import os
from os import path
from pathlib import Path
import sys
import tqdm
import numpy as np
import scipy.io as io
import scipy.ndimage as filters

import BVH as BVH
import Animation as Animation
from Quaternions import Quaternions
from Pivots import Pivots

## Directories
bvh_dir = "LAFAN1/bvh_valid/"
dest_dir = "LAFAN1/processed/"

def softmax(x, **kw):
    softness = kw.pop('softness', 1.0)
    maxi, mini = np.max(x, **kw), np.min(x, **kw)
    return maxi + np.log(softness + np.exp(mini - maxi))

def softmin(x, **kw):
    return -softmax(-x, **kw)

def process_file(filename, window=160, window_step=80):
    anim, names, frametime = BVH.load(filename)
    print(anim.shape)

    """ Convert to 60 fps """
    anim = anim[::2]

    """ Do FK """
    global_positions = Animation.positions_global(anim)

    """ Remove Unneeded Joints """
    positions = global_positions[:, np.array([
         0,
         1, 2, 3, 4,
         6, 7, 8, 9,
        11, 12, 13, 14,
        16, 17, 18, 19,
        20, 21])]

    """ Put on Floor """
    fid_l, fid_r = np.array([4, 5]), np.array([8, 9])
    foot_heights = np.minimum(positions[:, fid_l, 1], positions[:, fid_r, 1]).min(axis=1)
    floor_height = softmin(foot_heights, softness=0.5, axis=0)

    positions[:, :, 1] -= floor_height

    """ Add Reference Joint """
    trajectory_filterwidth = 3
    reference = positions[:, 0] * np.array([1, 0, 1])
    reference = filters.gaussian_filter1d(reference, trajectory_filterwidth, axis=0, mode='nearest')
    positions = np.concatenate([reference[:, np.newaxis], positions], axis=1)

    """ Get Foot Contacts """
    velfactor, heightfactor = np.array([0.05, 0.05]), np.array([3.0, 2.0])

    feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
    feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
    feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
    feet_l_h = positions[:-1, fid_l, 1]
    feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)).astype(float)

    feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
    feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
    feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
    feet_r_h = positions[:-1, fid_r, 1]
    feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)).astype(float)

    """ Get Root Velocity """
    velocity = (positions[1:, 0:1] - positions[:-1, 0:1]).copy()

    """ Remove Translation """
    positions[:, :, 0] = positions[:, :, 0] - positions[:, 0:1, 0]
    positions[:, :, 2] = positions[:, :, 2] - positions[:, 0:1, 2]

    """ Get Forward Direction """
    sdr_l, sdr_r, hip_l, hip_r = 12, 15, 1, 5
    across1 = positions[:, hip_l] - positions[:, hip_r]
    across0 = positions[:, sdr_l] - positions[:, sdr_r]
    across = across0 + across1
    across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]

    direction_filterwidth = 20
    forward = np.cross(across, np.array([[0, 1, 0]]))
    forward = filters.gaussian_filter1d(forward, direction_filterwidth, axis=0, mode='nearest')
    forward = forward / np.sqrt((forward ** 2).sum(axis=-1))[..., np.newaxis]

    """ Remove Y Rotation """
    target = np.array([[0, 0, 1]]).repeat(len(forward), axis=0)
    rotation = Quaternions.between(forward, target)[:, np.newaxis]
    positions = rotation * positions

    """ Get Root Rotation """
    velocity = rotation[1:] * velocity
    rvelocity = Pivots.from_quaternions(rotation[1:] * -rotation[:-1]).ps

    """ Add Velocity, RVelocity, Foot Contacts to vector """
    positions = positions[:-1]
    positions = positions.reshape(len(positions), -1)
    positions = np.concatenate([positions, velocity[:, :, 0]], axis=-1)
    positions = np.concatenate([positions, velocity[:, :, 2]], axis=-1)
    positions = np.concatenate([positions, rvelocity], axis=-1)
    #positions = np.concatenate([positions, feet_l, feet_r], axis=-1)

    """ Slide over windows """
    windows = []

    for j in range(0, len(positions) - window // 8, window_step):

        """ If slice too small pad out by repeating start and end poses """
        slice = positions[j:j + window]
        if len(slice) < window:
            left = slice[:1].repeat((window - len(slice)) // 2 + (window - len(slice)) % 2, axis=0)
            left[:, -7:-4] = 0.0
            right = slice[-1:].repeat((window - len(slice)) // 2, axis=0)
            right[:, -7:-4] = 0.0
            slice = np.concatenate([left, slice, right], axis=0)

        if len(slice) != window: raise Exception()

        windows.append(slice)

    return windows

def save_dataset(files, dest_dir):
    """Save the datapoints in 'data_csv' into three (speech, transcript, label) numpy arrays in 'save_dir'."""
    for i in tqdm.trange(len(files)):
        bvh = bvh_dir + files[i] + ".bvh"
        clips = process_file(bvh)

        if i == 0:
            X = clips
        else:
            X = np.concatenate((X, clips),  axis=0)

    x_save_path = path.join(dest_dir, f"valid.npy")
    np.save(x_save_path, X)
    print(f"Final dataset sizes: {X.shape}")

if __name__ == "__main__":
    #clips = process_file('C:/Users/vanta/Desktop/UnityDL/holden2015/dataset/LAFAN1/bvh/aiming1_subject1.bvh')
    #print(np.array(clips).shape)

    files = []
    p = Path(bvh_dir)
    print("Going to pre-process the following motion files:")
    files = sorted([i.stem for i in p.glob('**/*.bvh')])
    save_dataset(files, dest_dir)

