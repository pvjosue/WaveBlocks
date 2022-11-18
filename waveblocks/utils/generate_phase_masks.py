# Python imports
import math
from enum import Enum

# Third party libraries imports
import numpy as np
import torch


class PhaseMaskType(Enum):
    cubic = 1
    doublehelix = 2
    zernkie = 3


class PhaseMaskShape(Enum):
    round = 1
    square = 2


def create_phasemask(type, shape, dimensions, information):
    """
    Define the type of phase mask one want to generate e.g. cubic, zernike etc.
    type:
        - specifies the type of phase mask e.g. cubic, zernike
    shape:
        - specifies the shape of the phase mask e.g. round, square
    dimensions:
        - x: defines the x size of the output matrix
        - y: defines the y size of the output matrix
        - ratio: defines the scale of the phase mask e.g. 1 creates a bigger phasemask than 0.2
        - offset: allows to place the phase mask at a certain position in the matrix
    information:
        - max_phase_shift: specifies the min, max phase mask values
        - j: defines the zernike polynomial
    """

    if not isinstance(type, PhaseMaskType):
        raise TypeError("Type must be an instanace of PhaseMask Enum")

    if not isinstance(shape, PhaseMaskShape):
        raise TypeError("Shape must be an instance of PhaseMask Shape")

    # Extract dimension information of phase mask
    x = dimensions.get("x")
    y = dimensions.get("y")
    ratio = dimensions.get("ratio")
    offset = dimensions.get("offset")

    if x is None or y is None or ratio is None or offset is None:
        raise Exception("Missing dimension information")

    if information is None:
        raise Exception("Missing additional information to create phase mask")

    max_phase_shift = information.get("max_phase_shift")
    if max_phase_shift is None:
        raise Exception("Missing max_phase_shift for creation of cubic phase mask")

    if type == PhaseMaskType.cubic:
        """
        Creates a cubic phasemask
        """

        # Initialize standard phase mask
        X, Y = initStandardPhasemask(offset, [x, y])

        # Create a mask for the shape of the phase mask
        shape_mask = get_shape_mask(shape=shape, ratio=ratio, X=X, Y=Y)

        # Create cubic phase mask
        pm_img = max_phase_shift * shape_mask.float() * (torch.mul(torch.mul(X, X), X) + torch.mul(torch.mul(Y, Y), Y))

        if offset[0] >= 0 or offset[1] >= 0:
            pm_img = pm_img / pm_img.max() * max_phase_shift
        else:
            pm_img = pm_img / pm_img.min() * max_phase_shift

        return pm_img

    if type == PhaseMaskType.doublehelix:
        # Check for existance of standardInformation dictionary
        if information is None:
            raise Exception("Missing standard information")

        # TODO

    if type == PhaseMaskType.zernkie:
        """
        Creates a phasemask from a zernike polynomial
        """

        j = information.get("j")
        if j is None:
            raise Exception("Missing zernike index for creation of zernike phase mask")

        # Create pm_image for the output
        pm_img = np.zeros([x, y])  # , [('x',float),('y',float)])

        # Create zernike polynom for the specified size multiplied by the ratio
        zernike_x = math.floor(x * ratio)
        zernike_y = math.floor(y * ratio)
        zernike_matrix = np.ones((zernike_x, zernike_y))

        # Generate mask and apply on zernike matrix
        if shape == PhaseMaskShape.round:
            circular_mask = create_circular_mask(zernike_x, zernike_y)
            zernike_matrix = np.multiply(circular_mask, zernike_matrix)

        # Calcualte the value for the given n, m for a specific x, y coordinate
        for x_index in range(zernike_x):
            for y_index in range(zernike_y):

                if zernike_matrix[x_index, y_index] != 0:

                    shifted_x_index = x_index - math.ceil(zernike_x / 2)
                    shifted_y_index = y_index - math.ceil(zernike_y / 2)

                    r, theta = cartesian_to_radial(shifted_x_index, shifted_y_index)

                    # if not (tmp_x_offset < 0 or tmp_x_offset > zernike_x) and not (tmp_y_offset < 0 or tmp_y_offset > zernike_y):
                    zernike_matrix[x_index, y_index] = zernike_single_index(j, r, theta)

        # Place zernike according to offset
        x_offset = offset[0] * x
        y_offset = offset[1] * y

        for x_index in range(zernike_x):
            for y_index in range(zernike_y):
                # Calculate tmp offset
                tmp_x_offset = x_offset
                tmp_y_offset = y_offset

                x_val = int(np.floor(tmp_x_offset)) + int(np.floor(0.5 * x)) - int(np.floor(zernike_x / 2)) + x_index
                y_val = int(np.floor(tmp_y_offset)) + int(np.floor(0.5 * y)) - int(np.floor(zernike_y / 2)) + y_index

                # Place values of zernike
                if not (x_val < 0 or x_val >= x) and not (y_val < 0 or y_val >= y):
                    pm_img[x_val, y_val] = zernike_matrix[x_index, y_index]

        # Normalize values
        pm_img = (2 * max_phase_shift) * (pm_img - pm_img.min()) / (pm_img.max() - pm_img.min()) - max_phase_shift

        return torch.tensor(pm_img.astype(np.float32))


def create_cubic_mla_phasemask(dimensions, information):
    """
    Define the type of phase mask one want to generate e.g. cubic, zernike etc.
    type:
        - specifies the type of phase mask e.g. cubic, zernike
    shape:
        - specifies the shape of the phase mask e.g. round, square
    dimensions:
        - x: defines the x size of the output matrix
        - y: defines the y size of the output matrix
        - ratio: defines the scale of the phase mask e.g. 1 creates a bigger phasemask than 0.2
        - offset: allows to place the phase mask at a certain position in the matrix
    information:
        - max_phase_shift: specifies the min, max phase mask values
        - j: defines the zernike polynomial
    """

    """
    {
            "x": optic_config.pm_shape[0],
            "y": optic_config.pm_shape[1],
            "ratio": 0.4,  # 0.3,
            "offset": [0, 0],
        """
    real_x = dimensions["x"]
    real_y = dimensions["y"]
    dimensions["x"] = math.floor(dimensions["x"] / 3)
    dimensions["y"] = math.floor(dimensions["y"] / 3)
    mini_pm = create_phasemask(PhaseMaskType.cubic, PhaseMaskShape.square, dimensions, information)

    test = torch.zeros(real_x, real_y)

    x_size = mini_pm.shape[0]
    y_size = mini_pm.shape[1]
    for i in range(3):
        for j in range(3):
            xx = x_size * i
            yy = y_size * j
            test[xx : xx + x_size, yy : yy + y_size] = mini_pm

    return test


def create_circular_mask(h, w, center=None, radius=None):

    if center is None:  # use the middle of the image
        center = (int(w / 2), int(h / 2))
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w - center[0], h - center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    mask = dist_from_center <= radius
    return mask


def initStandardPhasemask(pm_offset, pm_shape):
    """
    Initialize a standard phase mask
    """

    u = np.arange(
        start=-0.5 + pm_offset[0],
        stop=0.5 + pm_offset[0],
        step=1 / pm_shape[0],
        dtype="float",
    )
    v = np.arange(
        start=-0.5 + pm_offset[1],
        stop=0.5 + pm_offset[1],
        step=1 / pm_shape[1],
        dtype="float",
    )
    X, Y = np.meshgrid(v, u)
    X = torch.from_numpy(X).float()
    Y = torch.from_numpy(Y).float()

    return X, Y


def get_shape_mask(shape, ratio, X, Y):
    """
    Create a mask for a specific shape
    """

    if shape == PhaseMaskShape.round:
        return ~(torch.sqrt(torch.mul(X, X) + torch.mul(Y, Y))).ge(torch.tensor([ratio]))
    if shape == PhaseMaskShape.square:
        return ~(torch.abs(X).ge(torch.tensor([ratio])) | torch.abs(Y).ge(torch.tensor([ratio])))
    else:
        raise TypeError("Missing shape information")


def cartesian_to_radial(x, y):
    """
    Transform cartesian coordinates to radial coordinates
    """

    r = math.sqrt(x ** 2 + y ** 2)

    theta = math.atan2(y, x)

    return r, theta


def zernike_single_index(j, radial_coordinate, theta):
    """
    Creates a zernike polynomial given j, radial_coordinate and theta
    """

    n = math.ceil((-3 + math.sqrt(9 + 8 * j)) / 2)

    m = 2 * j - n * (n + 2)

    return zernike(m, n, radial_coordinate, theta)


def zernike(m, n, radial_coordinate, theta):
    """
    Creates a zernike polynomial given m, n, radial_coordinate and theta
    """

    if n < 0:
        raise ValueError("The value of n must be a positive integer or zero")

    if not abs(m) <= n and not m % 2 == n % 2:
        raise ValueError("For a given n, m can only take on values -n, -n + 2, -n + 4, ...n. ")

    norm = normalization(m, n)

    zp = zernike_polynomial(m, n, radial_coordinate)

    if m >= 0:
        return norm * zp * math.cos(m * theta)

    else:
        return -norm * zp * math.sin(m * theta)


def zernike_polynomial(m, n, radial_coordinate):
    """
    Calculate zernike polynomial
    """

    value = 0
    m = abs(m)

    for s in range(int((n - m) / 2) + 1):

        value = value + (
            (-1) ** s * math.factorial(n - s) / (math.factorial(s) * math.factorial((n + m) / 2 - s) * math.factorial((n - m) / 2 - s))
        ) * radial_coordinate ** (n - 2 * s)

    return value


def normalization(m, n):
    """
    Normalization using kronecker formula
    """

    kroneckerDelta = 1 if m == 0 else 0

    norm = math.sqrt(2 * (n + 1) / (1 + kroneckerDelta))

    return norm


"""
Examples for creating cubic and zernike phase masks
matrix1 = create_phasemask(PhaseMaskType.cubic,
                           PhaseMaskShape.square,
                           {"x": 500, "y": 500, "ratio": 1.5, "offset": [1, 1]},
                           information = {"max_phase_shift": 100})
plt.imshow(matrix1.detach().numpy())
plt.colorbar()
plt.show()

matrix2 = create_phasemask(PhaseMaskType.cubic,
                           PhaseMaskShape.round,
                           {"x": 500, "y": 500, "ratio": 0.3, "offset": [0, 0]},
                           information = {"max_phase_shift": 1})
plt.imshow(matrix2.detach().numpy())
plt.colorbar()
plt.show()

matrix3 = create_phasemask(PhaseMaskType.zernkie,
                           PhaseMaskShape.round,
                           {"x": 500, "y": 500, "ratio": 0.5,
                               "offset": [0, 0]},
                           information={"j": 9, "max_phase_shift": 1})
plt.imshow(matrix3.detach().numpy())
plt.colorbar()
plt.show()

matrix4 = create_phasemask(PhaseMaskType.zernkie,
                           PhaseMaskShape.square,
                           {"x": 500, "y": 500, "ratio": 0.25,
                               "offset": [0, 0]},
                           information={"j": 9, "max_phase_shift": 2.5})
plt.imshow(matrix4.detach().numpy())
plt.colorbar()
plt.show()

from matplotlib import pyplot as plt
a = create_cubic_mla_phasemask(
    {
        "x": 251,
        "y": 251,
        "ratio": 0.4,  # 0.3,
        "offset": [0, 0],
    },
    information={"max_phase_shift": 5.4 * math.pi},
)
plt.imshow(a)
plt.show()
print("ok")
"""
