import numpy as np


def conv(image, kernel):
    """An implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    # For this assignment, we will use edge values to pad the images.
    # Zero padding will make derivatives at the image boundary very big,
    # whereas we want to ignore the edges at the boundary.
    pad_width0 = Hk // 2
    pad_width1 = Wk // 2
    pad_width = ((pad_width0, pad_width0), (pad_width1, pad_width1))
    padded = np.pad(image, pad_width, mode="edge")

    ### YOUR CODE HERE
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    for i in range(Hi):
        for j in range(Wi):
            out[i, j] = np.sum(padded[i : i + Hk, j : j + Wk] * kernel)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ### END YOUR CODE

    return out


def gaussian_kernel(size, sigma):
    """Implementation of Gaussian Kernel.

    This function follows the gaussian kernel formula,
    and creates a kernel matrix.

    Hints:
    - Use np.pi and np.exp to compute pi and exp.

    Args:
        size: int of the size of output matrix.
        sigma: float of sigma to calculate kernel.

    Returns:
        kernel: numpy array of shape (size, size).
    """

    kernel = np.zeros((size, size))

    ### YOUR CODE HERE
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    for i in range(size):
        for j in range(size):
            x = i - size // 2
            y = j - size // 2
            kernel[i, j] = (1 / (2 * np.pi * sigma**2)) * np.exp(
                -(x**2 + y**2) / (2 * sigma**2)
            )

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ### END YOUR CODE

    return kernel


def partial_x(img):
    """Computes partial x-derivative of input img.

    Hints:
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W).
    Returns:
        out: x-derivative image.
    """

    out = None

    ### YOUR CODE HERE
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    kernel = 1 / 2 * np.array([[-1, 0, 1]])
    out = conv(img, kernel)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ### END YOUR CODE

    return out


def partial_y(img):
    """Computes partial y-derivative of input img.

    Hints:
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W).
    Returns:
        out: y-derivative image.
    """

    out = None

    ### YOUR CODE HERE
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    kernel = 1 / 2 * np.array([[-1], [0], [1]])
    out = conv(img, kernel)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ### END YOUR CODE

    return out


def gradient(img):
    """Returns gradient magnitude and direction of input img.

    Args:
        img: Grayscale image. Numpy array of shape (H, W).

    Returns:
        G: Magnitude of gradient at each pixel in img.
            Numpy array of shape (H, W).
        theta: Direction(in degrees, 0 <= theta < 360) of gradient
            at each pixel in img. Numpy array of shape (H, W).

    Hints:
        - Use np.sqrt and np.arctan2 to calculate square root and arctan
    """
    G = np.zeros(img.shape)
    theta = np.zeros(img.shape)

    ### YOUR CODE HERE
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    Ix = partial_x(img)
    Iy = partial_y(img)
    G = np.sqrt(Ix**2 + Iy**2)
    theta = np.rad2deg(np.arctan2(Iy, Ix)) % 360

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ### END YOUR CODE

    return G, theta


def non_maximum_suppression(G, theta):
    """Performs non-maximum suppression.

    This function performs non-maximum suppression along the direction
    of gradient (theta) on the gradient magnitude image (G).

    Args:
        G: gradient magnitude image with shape of (H, W).
        theta: direction of gradients with shape of (H, W).

    Returns:
        out: non-maxima suppressed image.
    """
    H, W = G.shape
    out = np.zeros((H, W))

    # Round the gradient direction to the nearest 45 degrees
    theta = np.floor((theta + 22.5) / 45) * 45
    theta = (theta % 360.0).astype(np.int32)

    # print(G)
    ### BEGIN YOUR CODE
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    for i in range(1, H - 1):
        for j in range(1, W - 1):
            direction = theta[i, j]
            before, after = 0, 0
            if direction in [0, 180]:
                before, after = G[i, j - 1], G[i, j + 1]
            elif direction in [45, 225]:
                before, after = G[i - 1, j + 1], G[i + 1, j - 1]
            elif direction in [90, 270]:
                before, after = G[i - 1, j], G[i + 1, j]
            elif direction in [135, 315]:
                before, after = G[i - 1, j - 1], G[i + 1, j + 1]

            out[i, j] = G[i, j] if G[i, j] >= before and G[i, j] >= after else 0

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ### END YOUR CODE

    return out


def double_thresholding(img, high, low):
    """
    Args:
        img: numpy array of shape (H, W) representing NMS edge response.
        high: high threshold(float) for strong edges.
        low: low threshold(float) for weak edges.

    Returns:
        strong_edges: Boolean array representing strong edges.
            Strong edeges are the pixels with the values greater than
            the higher threshold.
        weak_edges: Boolean array representing weak edges.
            Weak edges are the pixels with the values smaller or equal to the
            higher threshold and greater than the lower threshold.
    """

    strong_edges = np.zeros(img.shape, dtype="bool")
    weak_edges = np.zeros(img.shape, dtype="bool")

    ### YOUR CODE HERE
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    strong_edges = img >= high
    weak_edges = (img >= low) & (img < high)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ### END YOUR CODE

    return strong_edges, weak_edges


def get_neighbors(y, x, H, W):
    """Return indices of valid neighbors of (y, x).

    Return indices of all the valid neighbors of (y, x) in an array of
    shape (H, W). An index (i, j) of a valid neighbor should satisfy
    the following:
        1. i >= 0 and i < H
        2. j >= 0 and j < W
        3. (i, j) != (y, x)

    Args:
        y, x: location of the pixel.
        H, W: size of the image.
    Returns:
        neighbors: list of indices of neighboring pixels [(i, j)].
    """
    neighbors = []

    for i in (y - 1, y, y + 1):
        for j in (x - 1, x, x + 1):
            if i >= 0 and i < H and j >= 0 and j < W:
                if i == y and j == x:
                    continue
                neighbors.append((i, j))

    return neighbors


def canny(img, kernel_size=5, sigma=1.4, high=20, low=15):
    """Implement canny edge detector by calling functions above.

    Args:
        img: binary image of shape (H, W).
        kernel_size: int of size for kernel matrix.
        sigma: float for calculating kernel.
        high: high threshold for strong edges.
        low: low threashold for weak edges.
    Returns:
        edge: numpy array of shape(H, W).
    """
    ### YOUR CODE HERE
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    gaussian_filter = gaussian_kernel(kernel_size, sigma)
    smoothed = conv(img, gaussian_filter)
    G, theta = gradient(smoothed)
    nms_result = non_maximum_suppression(G, theta)
    strong_edges, weak_edges = double_thresholding(nms_result, high, low)
    H, W = img.shape
    edge = np.zeros((H, W), dtype=bool)
    edge[strong_edges] = True

    for i in range(1, H - 1):
        for j in range(1, W - 1):
            if weak_edges[i, j] and edge[i, j] == False:
                neighbors = get_neighbors(i, j, H, W)
                for ni, nj in neighbors:
                    if edge[ni, nj] == True:
                        edge[i, j] = True
                        break

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ### END YOUR CODE

    return edge


def hough_transform(img):
    """Transform points in the input image into Hough space.

    Use the parameterization:
        rho = x * cos(theta) + y * sin(theta)
    to transform a point (x,y) to a sine-like function in Hough space.

    Args:
        img: binary image of shape (H, W).

    Returns:
        accumulator: numpy array of shape (m, n).
        rhos: numpy array of shape (m, ).
        thetas: numpy array of shape (n, ).
    """
    # Set rho and theta ranges
    W, H = img.shape
    diag_len = int(np.ceil(np.sqrt(W * W + H * H)))
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2 + 1)
    thetas = np.deg2rad(np.arange(-90.0, 90.0))

    # Cache some reusable values
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)

    # Initialize accumulator in the Hough space
    accumulator = np.zeros((2 * diag_len + 1, num_thetas), dtype=np.uint64)
    ys, xs = np.nonzero(img)

    # Transform each point (x, y) in image
    # Find rho corresponding to values in thetas
    # and increment the accumulator in the corresponding coordiate.
    ### YOUR CODE HERE
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    for i, theta in enumerate(thetas):
        for j in range(len(xs)):
            x, y = xs[j], ys[j]
            # Calculate rho for this (x,y) point and theta
            rho = x * cos_t[i] + y * sin_t[i]

            # Find the closest rho index
            rho_idx = np.argmin(np.abs(rhos - rho))

            # Increment the accumulator
            accumulator[rho_idx, i] += 1

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ### END YOUR CODE

    return accumulator, rhos, thetas
