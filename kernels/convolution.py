import numpy as np
from utils.image import pad_image


def correlation_func(
    input_f: np.array,
    kernel_h: np.array,
    padding: bool = False,
    padding_mode: str = None,
) -> np.array:
    """
    Correlation function

    input_f: 2D array, input array
    kernel_h: 2D array, kernel array
    padding: boolean, True/False, padding or not
    padding_mode: zero/replicator/reflection

    Output: output_g, 2D array
    """
    kn, km = kernel_h.shape
    padded, output_g, r1, r2, h_range, w_range = pad_image(
        input_f, kernel_h.shape, padding, padding_mode
    )

    for i in h_range:
        for j in w_range:
            if padding:
                window = padded[i - r1 : i + r1 + 1, j - r2 : j + r2 + 1]
                output_g[i - r1, j - r2] = np.sum(window * kernel_h)
            else:
                window = padded[i : i + kn, j : j + km]
                output_g[i, j] = np.sum(window * kernel_h)

    return output_g


def convolution_func_v1(
    input_f: np.array,
    kernel_h: np.array,
    padding: bool = False,
    padding_mode: str = None,
) -> np.array:
    """
    Convolution function

    input_f: 2D array, input array
    kernel_h: 2D array, kernel array
    padding: boolean, True/False, padding or not
    padding_mode: zero/replicator/reflection

    Output: output_g, 2D array
    """
    kn, km = kernel_h.shape
    padded, output_g, r1, r2, h_range, w_range = pad_image(
        input_f, kernel_h.shape, padding, padding_mode
    )

    kernel_flipped = np.flip(kernel_h)

    for i in h_range:
        for j in w_range:
            if padding:
                window = padded[i - r1 : i + r1 + 1, j - r2 : j + r2 + 1]
                output_g[i - r1, j - r2] = np.sum(window * kernel_flipped)
            else:
                window = padded[i : i + kn, j : j + km]
                output_g[i, j] = np.sum(window * kernel_flipped)

    return output_g


def convolution_func_v2(
    input_f: np.array,
    kernel_h: np.array,
    padding: bool = False,
    padding_mode: str = None,
) -> np.array:
    """
    Convolution function

    input_f: 2D array, input array
    kernel_h: 2D array, kernel array
    padding: boolean, True/False, padding or not
    padding_mode: zero/replicator/reflection

    Output: output_g, 2D array
    """
    # Implement using by calling correlation_func
    kernel_flipped = np.flip(kernel_h)
    output_g = correlation_func(input_f, kernel_flipped, padding, padding_mode)
    return output_g
