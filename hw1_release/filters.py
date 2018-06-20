import numpy as np


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    temp_m = np.zeros((Hi+Hk-1, Wi+Wk-1))

    for i in range(Hi+Hk-1):
        for j in range(Wi+Wk-1):
            temp = 0
            for m in range(Hk):
                for n in range(Wk):
                    if ((i-m)>=0 and (i-m)<Hi and (j-n)>=0 and (j-n)<Wi):
                        temp += image[i-m][j-n] * kernel[m][n]
            temp_m[i][j] = temp
            
    for i in range(Hi):
        for j in range(Wi):
            out[i][j] = temp_m[int(i + (Hk - 1)/2)][int(j + (Wk - 1)/2)]

    return out

def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W)
        pad_width: width of the zero padding (left and right padding)
        pad_height: height of the zero padding (bottom and top padding)

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width)
    """

    H, W = image.shape
    out = np.zeros((H + 2*pad_height, W + 2*pad_width))

    out[pad_height:pad_height + H, pad_width:pad_width + W] = image

    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    image_padding = zero_pad(image, Hk // 2, Wk // 2)
    kernel_flip = np.flip(np.flip(kernel, 0), 1)

    for i in range(Hi):
        for j in range(Wi):            
            out[i][j] = np.sum(np.multiply(kernel_flip, image_padding[i:(i+Hk), j:(j+Wk)]))

    return out

def conv_faster(image, kernel):
    """
    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    return out

def cross_correlation(f, g):
    """ Cross-correlation of f and g

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)

    Returns:
        out: numpy array of shape (Hf, Wf)
    """
    
    out = conv_fast(f, np.flip(np.flip(g, 0), 1))

    return out

def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of f and g

    Subtract the mean of g from g so that its mean becomes zero

    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)

    Returns:
        out: numpy array of shape (Hf, Wf)
    """
    
    out = conv_fast(f, np.flip(np.flip(g - np.mean(g), 0), 1))

    return out

def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of f and g

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)

    Returns:
        out: numpy array of shape (Hf, Wf)
    """

    Hi, Wi = f.shape
    Hk, Wk = g.shape
    out = np.zeros((Hi, Wi))

    f_padding = zero_pad(f, Hk // 2, Wk // 2)
    g_norm = (g - np.mean(g))/np.std(g)

    for i in range(Hi):
        for j in range(Wi):
            padding_norm = np.array(f_padding[i:(i+Hk), j:(j+Wk)])
            padding_norm = (padding_norm - np.mean(padding_norm))/np.std(padding_norm)
            out[i][j] = np.sum(np.multiply(g_norm, padding_norm))

    return out
