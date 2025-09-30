"""
Copyright 2025, European Space Agency (ESA)
Licensed under ESA Software Community Licence Permissive (Type 3) - v2.4
"""

import numpy as np
import numpy.typing as npt
import scipy


def kernel_generation(shape: tuple[int, ...]) -> np.ndarray:
    """
    Generates a kernel with the given shape, where each element is the reciprocal of the product of the shape dimensions.

    Parameters:
    shape (tuple[int, ...]): The shape of the kernel.

    Returns:
    np.ndarray: A numpy array representing the kernel.
    """
    return np.full(shape, 1 / np.prod(shape))


def single_baseline_single_pol_coh(primary_complex: npt.NDArray[complex],
                                   secondary_complex: npt.NDArray[complex],
                                   avg_kernel_shape: tuple[int, ...],
                                   ) -> npt.NDArray[complex]:
    """
    Compute the coherence map (complex).

    The coherence map at an azimuth/range pixel (a, r) is defined as:

                                E[S(a, r) * conj(P(a, r))]
       Coh{P, S}(a, r) :=  -----------------------------------
                            sqrt(Var[P(a, r)] * Var[S(a, r)])

    Parameters:
    primary_complex (npt.NDArray[complex]): Complex array of the primary image.
    secondary_complex (npt.NDArray[complex]): Complex array of the secondary image.
    avg_kernel_shape (tuple[int, ...]): Shape of the averaging kernel.
    flag_avg (bool): If False, only the Hermitian product is applied without averaging. Default is True.

    Returns:
    npt.NDArray[complex]: The [Nazm x Nrng] coherence map.

    Raises:
    ValueError: If the shapes of primary_complex and secondary_complex do not match.
    """
    if primary_complex.shape != secondary_complex.shape:
        raise ValueError(f"Coh inputs have different shapes {primary_complex.shape} != {secondary_complex.shape}")

    kernel = kernel_generation(avg_kernel_shape)

    covariance_primary_secondary = (primary_complex * np.conj(secondary_complex)).astype(np.complex64)
    covariance_primary_secondary = scipy.signal.convolve2d(covariance_primary_secondary,
                                                            kernel,
                                                            boundary="symm",
                                                            mode="same")

    variance_primary = np.abs(primary_complex)**2
    variance_primary = scipy.signal.convolve2d(variance_primary,
                                                kernel,
                                                boundary="symm",
                                                mode="same")

    variance_secondary = np.abs(secondary_complex)**2
    variance_secondary = scipy.signal.convolve2d(variance_secondary,
                                                    kernel,
                                                    boundary="symm",
                                                    mode="same")

    variance_primary_variance_secondary = variance_primary * variance_secondary
    valid = variance_primary_variance_secondary > 0.0

    coherence = np.empty_like(covariance_primary_secondary)
    coherence[valid] = covariance_primary_secondary[valid] / np.sqrt(
        variance_primary_variance_secondary[valid]
    )

    coherence[~valid] = 0
    coherence[np.isnan(coherence)] = 0 + 0j



    return coherence
'''

import numpy as np
import scipy
from typing import Tuple
from numpy import ndarray
import numpy.typing as npt

def kernel_generation(shape: Tuple[int, ...]) -> np.ndarray:
    """
    Generates a kernel with the given shape, where each element is the reciprocal of the product of the shape dimensions.

    Parameters:
    shape (tuple[int, ...]): The shape of the kernel.

    Returns:
    np.ndarray: A numpy array representing the kernel.
    """
    return np.full(shape, 1 / np.prod(shape))


def single_baseline_single_pol_coh(primary_complex: npt.NDArray[complex],
                                   secondary_complex: npt.NDArray[complex],
                                   avg_kernel_shape: Tuple[int, ...],
                                   ) -> npt.NDArray[complex]:
    """
    Compute the coherence map (complex).

    The coherence map at an azimuth/range pixel (a, r) is defined as:

                                E[S(a, r) * conj(P(a, r))]
       Coh{P, S}(a, r) :=  -----------------------------------
                            sqrt(Var[P(a, r)] * Var[S(a, r)])

    Parameters:
    primary_complex (npt.NDArray[complex]): Complex array of the primary image.
    secondary_complex (npt.NDArray[complex]): Complex array of the secondary image.
    avg_kernel_shape (tuple[int, ...]): Shape of the averaging kernel.
    flag_avg (bool): If False, only the Hermitian product is applied without averaging. Default is True.

    Returns:
    npt.NDArray[complex]: The [Nazm x Nrng] coherence map.

    Raises:
    ValueError: If the shapes of primary_complex and secondary_complex do not match.
    """
    if primary_complex.shape != secondary_complex.shape:
        raise ValueError(f"Coh inputs have different shapes {primary_complex.shape} != {secondary_complex.shape}")

    kernel = kernel_generation(avg_kernel_shape)

    covariance_primary_secondary = (primary_complex * np.conj(secondary_complex)).astype(np.complex64)
    covariance_primary_secondary = scipy.signal.convolve2d(covariance_primary_secondary,
                                                            kernel,
                                                            boundary="symm",
                                                            mode="same")

    variance_primary = np.abs(primary_complex)**2
    variance_primary = scipy.signal.convolve2d(variance_primary,
                                                kernel,
                                                boundary="symm",
                                                mode="same")

    variance_secondary = np.abs(secondary_complex)**2
    variance_secondary = scipy.signal.convolve2d(variance_secondary,
                                                    kernel,
                                                    boundary="symm",
                                                    mode="same")

    variance_primary_variance_secondary = variance_primary * variance_secondary
    valid = variance_primary_variance_secondary > 0.0

    coherence = np.empty_like(covariance_primary_secondary)
    coherence[valid] = covariance_primary_secondary[valid] / np.sqrt(
        variance_primary_variance_secondary[valid]
    )

    coherence[~valid] = 0
    coherence[np.isnan(coherence)] = 0 + 0j



    return coherence

'''

