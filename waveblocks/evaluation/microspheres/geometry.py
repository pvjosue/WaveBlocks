"""
Extracted the shpere implemention from the package pymrt since the package had errors

Link: https://pypi.org/project/pymrt/
Repository: https://bitbucket.org/norok2/pymrt/src/master/
"""

# Third party libraries
import numpy as np

# Waveblocks imports
import waveblocks.evaluation.microspheres as wbm


def sphere(shape, radius, position=0.5):
    """
    Generate a mask whose shape is a sphere.

    Args:
        shape (int|iterable[int]): The shape of the mask in px.
        radius (float): The radius of the sphere in px.
        position (float|iterable[float]): The position of the center.
            Values are relative to the lowest edge.
            They are interpreted as relative to the shape.

    Returns:
        mask (np.ndarray): Array of boolean describing the geometrical object.

    Examples:
        >>> sphere(3, 1)
        array([[[False, False, False],
                [False,  True, False],
                [False, False, False]],
        <BLANKLINE>
               [[False,  True, False],
                [ True,  True,  True],
                [False,  True, False]],
        <BLANKLINE>
               [[False, False, False],
                [False,  True, False],
                [False, False, False]]], dtype=bool)
        >>> sphere(5, 2)
        array([[[False, False, False, False, False],
                [False, False, False, False, False],
                [False, False,  True, False, False],
                [False, False, False, False, False],
                [False, False, False, False, False]],
        <BLANKLINE>
               [[False, False, False, False, False],
                [False,  True,  True,  True, False],
                [False,  True,  True,  True, False],
                [False,  True,  True,  True, False],
                [False, False, False, False, False]],
        <BLANKLINE>
               [[False, False,  True, False, False],
                [False,  True,  True,  True, False],
                [ True,  True,  True,  True,  True],
                [False,  True,  True,  True, False],
                [False, False,  True, False, False]],
        <BLANKLINE>
               [[False, False, False, False, False],
                [False,  True,  True,  True, False],
                [False,  True,  True,  True, False],
                [False,  True,  True,  True, False],
                [False, False, False, False, False]],
        <BLANKLINE>
               [[False, False, False, False, False],
                [False, False, False, False, False],
                [False, False,  True, False, False],
                [False, False, False, False, False],
                [False, False, False, False, False]]], dtype=bool)
    """
    return nd_superellipsoid(
        shape, radius, 2.0, position, 3, rel_position=True, rel_sizes=False
    )


def nd_superellipsoid(
    shape,
    semisizes=0.5,
    indexes=2,
    position=0.5,
    n_dim=None,
    rel_position=True,
    rel_sizes=True,
):
    """
    Generate a mask whose shape is an N-dim superellipsoid.

    The cartesian equations are:

    .. math::
        \\sum[\\abs(\\frac{x_n}{a_n})^{k_n}] < 1.0

    where :math:`n` runs through the dims, :math:`x` are the cartesian
    coordinate, :math:`a` are the semi-sizes (semi-axes) and
    :math:`k` are the indexes.

    When the index is 2, an ellipsis is generated.

    Args:
        shape (int|iterable[int]): The shape of the mask in px.
        semisizes (float|iterable[float]): The N-dim superellipsoid axes sizes.
            The values interpretation depend on `rel_sizes`.
        position (float|iterable[float]): The position of the center.
            Values are relative to the lowest edge.
            The values interpretation depend on `rel_position`.
        indexes (float|tuple[float]): The exponent of the summed terms.
            If 2, generates n-dim ellipsoids.
        n_dim (int|None): The number of dimensions.
            If None, the number of dims is guessed from the other parameters.
        rel_position (bool): Interpret positions as relative values.
            If True, position values are interpreted as relative,
            i.e. they are scaled for `shape` using `mrt.utils.grid_coord()`.
            Otherwise, they are interpreted as absolute (in px).
        rel_sizes (bool): Interpret sizes as relative values.
            If True, `semisizes` values are interpreted as relative,
            i.e. they are scaled for `shape` using `mrt.utils.grid_coord()`.
            Otherwise, they are interpreted as absolute (in px).

    Returns:
        mask (np.ndarray): Array of boolean describing the geometrical object.
    """
    if not n_dim:
        n_dim = wbm.combine_iter_len((shape, position, semisizes, indexes))
    # check compatibility of given parameters
    shape = wbm.auto_repeat(shape, n_dim, check=True)
    position = wbm.auto_repeat(position, n_dim, check=True)
    semisizes = wbm.auto_repeat(semisizes, n_dim, check=True)
    indexes = wbm.auto_repeat(indexes, n_dim, check=True)
    # get correct position
    if rel_sizes:
        semisizes = rel2abs(shape, semisizes)
    position = wbm.grid_coord(shape, position, is_relative=rel_position, use_int=False)
    # create the mask
    mask = np.zeros(shape, dtype=float)
    for x_i, semiaxis, index in zip(position, semisizes, indexes):
        mask += np.abs(x_i / semiaxis) ** index

    return mask <= 1.0


def rel2abs(shape, size=0.5):
    """
    Calculate the absolute size from a relative size for a given shape.

    Args:
        shape (int|iterable[int]): The shape of the mask in px.
        size (float|tuple[float]): Relative position (to the lowest edge).
            Each element of the tuple should be in the range [0, 1].

    Returns:
        position (float|tuple[float]): Absolute position inside the shape.
            Each element of the tuple should be in the range [0, dim - 1],
            where dim is the corresponding dimension of the shape.

    Examples:
        >>> rel2abs((100, 100, 101, 101), (0.0, 1.0, 0.0, 1.0))
        (0.0, 99.0, 0.0, 100.0)
        >>> rel2abs((100, 99, 101))
        (49.5, 49.0, 50.0)
        >>> rel2abs((100, 200, 50, 99, 37), (0.0, 1.0, 0.2, 0.3, 0.4))
        (0.0, 199.0, 9.8, 29.4, 14.4)
        >>> rel2abs((100, 100, 100), (1.0, 10.0, -1.0))
        (99.0, 990.0, -99.0)
        >>> shape = (100, 100, 100, 100, 100)
        >>> abs2rel(shape, rel2abs(shape, (0.0, 0.25, 0.5, 0.75, 1.0)))
        (0.0, 0.25, 0.5, 0.75, 1.0)
    """
    size = wbm.auto_repeat(size, len(shape), check=True)
    return tuple((s - 1.0) * p for p, s in zip(size, shape))
