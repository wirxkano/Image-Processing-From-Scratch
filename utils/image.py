import numpy as np


def pad_image(
    input_f: np.array, kernel_shape: tuple[int, int], padding: bool, padding_mode: str
):
    """Helper: pad image and compute ranges."""
    H, W = input_f.shape
    kh, kw = kernel_shape
    r1, r2 = kh // 2, kw // 2

    if padding:
        if padding_mode == "zero":
            mode = "constant"
        elif padding_mode == "replicator":
            mode = "edge"
        elif padding_mode == "reflection":
            mode = "symmetric"
        else:
            raise ValueError(f"Unknown padding mode {padding_mode}")

        padded = np.pad(input_f, mode=mode, pad_width=((r1, r1), (r2, r2)))
        output = np.zeros((H, W))
        h_range = range(r1, H + r1)
        w_range = range(r2, W + r2)
    else:
        padded = input_f
        output = np.zeros((H - kh + 1, W - kw + 1))
        h_range = range(H - kh + 1)
        w_range = range(W - kw + 1)

    return padded, output, r1, r2, h_range, w_range
