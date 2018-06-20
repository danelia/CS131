import numpy as np

def conv(image, kernel):
    """ An implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    # For this assignment, we will use edge values to pad the images.
    # Zero padding will make derivatives at the image boundary very big,
    # whereas we want to ignore the edges at the boundary.
    pad_width0 = Hk // 2
    pad_width1 = Wk // 2
    pad_width = ((pad_width0,pad_width0),(pad_width1,pad_width1))
    padded = np.pad(image, pad_width, mode='edge')

    kh = int((Hk - 1) / 2)
    kw = int((Wk - 1) / 2)
    for i in range(0, Hi):
        for j in range(0, Wi):
            out[i][j] = np.sum(
                kernel * padded[pad_width0 + i - kh:pad_width0 + i + kh + 1,
                         pad_width1 + j - kw : pad_width1 + j + kw + 1])

    return out

def gaussian_kernel(size, sigma):
    """ Implementation of Gaussian Kernel.
    
    This function follows the gaussian kernel formula,
    and creates a kernel matrix.

    Hints:
    - Use np.pi and np.exp to compute pi and exp
    
    Args:
        size: int of the size of output matrix
        sigma: float of sigma to calculate kernel

    Returns:
        kernel: numpy array of shape (size, size)
    """  
    
    kernel = np.zeros((size, size))

    for i in range(size):
        for j in range(size):
            kernel[i][j] = (1.0 /(2.0 * np.pi * sigma**2)) * \
                np.exp(((-1.0) * ((i-((size - 1) / 2))**2 + (j-(size - 1) / 2)**2))/(2 * (sigma ** 2)))

    return kernel

def partial_x(img):
    """ Computes partial x-derivative of input img.

    Hints: 
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W)
    Returns:
        out: x-derivative image
    """

    out = conv(img, np.array([-0.5, 0, 0.5]).reshape(1, 3))

    return out

def partial_y(img):
    """ Computes partial y-derivative of input img.

    Hints: 
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W)
    Returns:
        out: y-derivative image
    """
    
    out = conv(img, np.array([-0.5, 0, 0.5]).reshape(3, 1))

    return out

def gradient(img):
    """ Returns gradient magnitude and direction of input img.

    Args:
        img: Grayscale image. Numpy array of shape (H, W)

    Returns:
        G: Magnitude of gradient at each pixel in img.
            Numpy array of shape (H, W)
        theta: Direction(in degrees, 0 <= theta < 360) of gradient
            at each pixel in img. Numpy array of shape (H, W)
    """
    G = np.zeros(img.shape)
    theta = np.zeros(img.shape)

    G_x = partial_x(img)
    G_y = partial_y(img)
    
    G = np.sqrt(np.add(np.square(G_x), np.square(G_y)))
    theta = (np.arctan2(G_y, G_x) * 180 / np.pi + 180) % 360
    
    return G, theta


def non_maximum_suppression(G, theta):
    """ Performs non-maximum suppression

    This function performs non-maximum suppression along the direction
    of gradient (theta) on the gradient magnitude image (G).
    
    Args:
        G: gradient magnitude image with shape of (H, W)
        theta: direction of gradients with shape of (H, W)

    Returns:
        out: non-maxima suppressed image
    """
    H, W = G.shape
    out = np.zeros((H, W))

    # Round the gradient direction to the nearest 45 degrees
    theta = np.floor((theta + 22.5) / 45) * 45 % 360
    
    for i in range(1, H-1):
        for j in range(1,W-1):
            curr = theta[i,j]
            if curr == 0 or curr == 180:
                if G[i, j] >= np.max([G[i, j-1], G[i, j+1]]):
                    out[i,j] = G[i,j]
                    continue
            elif curr == 45 or curr == 225:
                if G[i, j] >= np.max([G[i-1, j-1], G[i+1, j+1]]):
                    out[i,j] = G[i,j]
                    continue
            elif curr == 90 or curr == 270:
                if G[i, j] >= np.max([G[i-1, j], G[i+1, j]]):
                    out[i,j] = G[i,j]
                    continue
            elif curr == 135 or curr == 315:
                if G[i, j] >= np.max([G[i-1, j+1], G[i+1, j-1]]):
                    out[i,j] = G[i,j]
                    continue
            
            out[i, j] = 0 # unnecessary, just to be safe

    return out

def double_thresholding(img, high, low):
    """
    Args:
        img: numpy array of shape (H, W) representing NMS edge response
        high: high threshold(float) for strong edges
        low: low threshold(float) for weak edges

    Returns:
        strong_edges: Boolean array representing strong edges.
            Strong edeges are the pixels with the values above
            the higher threshold.
        weak_edges: Boolean array representing weak edges.
            Weak edges are the pixels with the values below the
            higher threshould and above the lower threshold.
    """

    strong_edges = np.zeros(img.shape)
    weak_edges = np.zeros(img.shape)
    
    strong_edges = strong_edges.astype(bool)
    weak_edges = weak_edges.astype(bool)
    
    H, W = img.shape
    for i in range(0, H):
        for j in range(0, W):
            if img[i, j] > high:
                strong_edges[i, j] = 1
                weak_edges[i, j] = 0
            elif img[i, j] > low and img[i, j] <= high:
                strong_edges[i, j] = 0
                weak_edges[i, j] = 1
            else:
                strong_edges[i, j] = 0
                weak_edges[i, j] = 0
    
    return strong_edges, weak_edges


def get_neighbors(y, x, H, W):
    """ Return indices of valid neighbors of (y, x)

    Return indices of all the valid neighbors of (y, x) in an array of
    shape (H, W). An index (i, j) of a valid neighbor should satisfy
    the following:
        1. i >= 0 and i < H
        2. j >= 0 and j < W
        3. (i, j) != (y, x)

    Args:
        y, x: location of the pixel
        H, W: size of the image
    Returns:
        neighbors: list of indices of neighboring pixels [(i, j)]
    """
    neighbors = []

    for i in (y-1, y, y+1):
        for j in (x-1, x, x+1):
            if i >= 0 and i < H and j >= 0 and j < W:
                if (i == y and j == x):
                    continue
                neighbors.append((i, j))

    return neighbors

def link_edges(strong_edges, weak_edges):
    """ Find weak edges connected to strong edges and link them.

    Iterate over each pixel in strong_edges and perform breadth first
    search across the connected pixels in weak_edges to link them.
    Here we consider a pixel (a, b) is connected to a pixel (c, d)
    if (a, b) is one of the eight neighboring pixels of (c, d).

    Args:
        strong_edges: binary image of shape (H, W)
        weak_edges: binary image of shape (H, W)
    Returns:
        edges: numpy array of shape(H, W)
    """

    H, W = strong_edges.shape
    indices = np.stack(np.nonzero(strong_edges)).T
    edges = np.zeros((H, W))
    
    for i, j in indices:
        edges[i][j] = 1
        
        visited = np.zeros(weak_edges.shape)
        queue = []
        queue.append((i, j))
        visited[i][j] = 1
        while len(queue):
            m, n = queue.pop()
            for u, v in get_neighbors(m, n, H, W):
                if weak_edges[u][v] and not visited[u][v]:
                    queue.append((u, v))
                    visited[u][v] = 1
                    edges[u][v] = 1
                    
    return edges

def canny(img, kernel_size=5, sigma=1.4, high=20, low=15):
    """ Implement canny edge detector by calling functions above.

    Args:
        img: binary image of shape (H, W)
        kernel_size: int of size for kernel matrix
        sigma: float for calculating kernel
        high: high threshold for strong edges
        low: low threashold for weak edges
    Returns:
        edge: numpy array of shape(H, W)
    """
    
    kernel = gaussian_kernel(kernel_size, sigma)
    smoothed = conv(img, kernel)
    G, theta = gradient(smoothed)
    non_maxima_img = non_maximum_suppression(G, theta)
    strong_edges, weak_edges = double_thresholding(non_maxima_img, high, low)
    edge = link_edges(strong_edges, weak_edges)

    return edge


def hough_transform(img):
    """ Transform points in the input image into Hough space.

    Use the parameterization:
        rho = x * cos(theta) + y * sin(theta)
    to transform a point (x,y) to a sine-like function in Hough space.

    Args:
        img: binary image of shape (H, W)
        
    Returns:
        accumulator: numpy array of shape (m, n)
        rhos: numpy array of shape (m, )
        thetas: numpy array of shape (n, )
    """
    # Set rho and theta ranges
    W, H = img.shape
    diag_len = int(np.ceil(np.sqrt(W * W + H * H)))
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2.0 + 1)
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
    for y, x in zip(ys, xs):
        for i in range(num_thetas):
            curr_rho = x * cos_t[i] + y * sin_t[i]
            accumulator[int(curr_rho + diag_len), i] += 1

    return accumulator, rhos, thetas