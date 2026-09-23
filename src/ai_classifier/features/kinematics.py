"""Kinematic and trajectory feature engineering for badminton stroke recognition."""

from __future__ import annotations

import numpy as np


def resample(array: np.ndarray, length: int) -> np.ndarray:
    """Resample a 2D temporal sequence to a fixed length using linear interpolation."""
    values = np.nan_to_num(array.astype(np.float32), nan=0.0).reshape(len(array), -1)
    if len(values) == length:
        return values
    source = np.linspace(0.0, 1.0, len(values))
    target = np.linspace(0.0, 1.0, length)
    return np.stack(
        [np.interp(target, source, values[:, index]) for index in range(values.shape[1])],
        axis=1,
    ).astype(np.float32)


def structured_feature_arrays(
    joints: np.ndarray,
    position: np.ndarray,
    shuttle: np.ndarray,
    length: int,
) -> np.ndarray:
    """Build structured feature vector from joints, court position, and shuttle trajectory.
    
    Includes coordinate sequences, first-order velocity derivatives (mean and std).
    """
    arrays = [resample(array, length) for array in (joints, position, shuttle)]
    sequence = np.concatenate(arrays, axis=1)
    velocity = np.diff(sequence, axis=0, prepend=sequence[:1])
    return np.concatenate([sequence.ravel(), velocity.mean(0), velocity.std(0)]).astype(np.float32)


def target_aligned_feature_arrays(
    joints: np.ndarray,
    position: np.ndarray,
    shuttle: np.ndarray,
    target_index: int,
    length: int,
    before: int = 15,
    after: int = 30,
) -> np.ndarray:
    """Extract kinematic feature vector centered around the hit impact target frame.
    
    Extracts a temporal window [-before, +after] around target_index, resamples to `length`,
    and computes displacement velocity and variation statistics.
    """
    window_length = before + 1 + after
    arrays = []
    for array in (joints, position, shuttle):
        safe_target = min(max(int(target_index), 0), max(len(array) - 1, 0))
        output = np.zeros((window_length, *array.shape[1:]), dtype=np.float32)
        source_start = max(0, safe_target - before)
        source_end = min(len(array), safe_target + after + 1)
        dest_start = before - min(before, safe_target)
        span = min(source_end - source_start, window_length - dest_start)
        if span > 0:
            output[dest_start:dest_start + span] = array[source_start:source_start + span]
        arrays.append(output)

    modalities = [resample(array, length) for array in arrays]
    target = np.zeros((window_length, 1), dtype=np.float32)
    target[before] = 1.0
    modalities.append(resample(target, length))

    sequence = np.concatenate(modalities, axis=1)
    velocity = np.diff(sequence[:, :-1], axis=0, prepend=sequence[:1, :-1])
    return np.concatenate([sequence.ravel(), velocity.mean(0), velocity.std(0)]).astype(np.float32)

