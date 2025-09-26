import numpy as np
from utils.image import pad_image


def nonlinear_filtering(
    input_f: np.array,
    kernel_h: int,
    kernel_w: int,
    nl_func: str = "median",
    padding: bool = False,
    padding_mode: str = None,
) -> np.array:
    """
    Do Nonlinear (median/min/max) filtering

    input_f: 2D array, input array
    kernel_h, kernel_w: int values, height and width of kernel
    nl_func: string, median/min/max function
    padding: boolean, True/False, padding or not
    padding_mode: zero/replicator/reflection

    Output: output_g, 2D array
    """
    padded, output_g, r1, r2, h_range, w_range = pad_image(
        input_f, (kernel_h, kernel_w), padding, padding_mode
    )

    np_func = {"median": np.median, "min": np.min, "max": np.max}

    for i in h_range:
        for j in w_range:
            if padding:
                window = padded[i - r1 : i + r1 + 1, j - r2 : j + r2 + 1]
                value = np_func[nl_func](window)
                output_g[i - r1, j - r2] = value
            else:
                window = padded[i : i + kernel_h, j : j + kernel_w]
                value = np_func[nl_func](window)
                output_g[i, j] = value

    return output_g
