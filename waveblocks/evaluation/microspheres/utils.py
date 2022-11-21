"""
Extracted the shpere implemention from the package pymrt since the package had errors

Link: https://pypi.org/project/pymrt/
Repository: https://bitbucket.org/norok2/pymrt/src/master/

"""

# Third party libraries
import numpy as np


def combine_iter_len(items, combine=max):
    """
    Determine the maximum length of an item within a collection of items.

    Args:
        items (iterable): The collection of items to inspect.
        combine (callable): The combination method.

    Returns:
        num (int): The combined length of the collection.
            If none of the items are iterable, the result is `1`.

    Examples:
        >>> a = list(range(10))
        >>> b = tuple(range(5))
        >>> c = set(range(20))
        >>> combine_iter_len((a, b, c))
        20
        >>> combine_iter_len((a, b, c), min)
        5
        >>> combine_iter_len((1, a))
        10
    """
    num = None
    for val in items:
        try:
            iter(val)
        except TypeError:
            pass
        else:
            if num is None:
                num = len(val)
            else:
                num = combine(len(val), num)
    if num is None:
        num = 1
    return num


# ======================================================================
def auto_repeat(obj, n, force=False, check=False):
    """
    Automatically repeat the specified object n times.

    If the object is not iterable, a tuple with the specified size is returned.
    If the object is iterable, the object is left untouched.

    Args:
        obj: The object to operate with.
        n (int): The length of the output object.
        force (bool): Force the repetition, even if the object is iterable.
        check (bool): Ensure that the object has length n.

    Returns:
        val (tuple): Returns obj repeated n times.

    Raises:
        AssertionError: If force is True and the object does not have length n.

    Examples:
        >>> auto_repeat(1, 3)
        (1, 1, 1)
        >>> auto_repeat([1], 3)
        [1]
        >>> auto_repeat([1, 3], 2)
        [1, 3]
        >>> auto_repeat([1, 3], 2, True)
        ([1, 3], [1, 3])
        >>> auto_repeat([1, 2, 3], 2, True, True)
        ([1, 2, 3], [1, 2, 3])
        >>> auto_repeat([1, 2, 3], 2, False, True)
        Traceback (most recent call last):
            ...
        AssertionError
    """
    try:
        iter(obj)
    except TypeError:
        force = True
    finally:
        if force:
            obj = (obj,) * n
    if check:
        assert len(obj) == n
    return obj


# ======================================================================
def grid_coord(shape, position=0.5, is_relative=True, use_int=True, dense=False):
    """
    Calculate the generic x_i coordinates for N-dim operations.

    Args:
        shape (iterable[int]): The shape of the mask in px.
        position (float|iterable[float]): Relative position of the origin.
            Values are in the [0, 1] interval.
        is_relative (bool): Interpret origin as relative.
        dense (bool): Determine the shape of the mesh-grid arrays.
        use_int (bool): Force interger values for the coordinates.

    Returns:
        coord (list[np.ndarray]): mesh-grid ndarrays.
            The shape is identical if dense is True, otherwise only one
            dimension is larger than 1.

    Examples:
        >>> grid_coord((4, 4))
        [array([[-2],
               [-1],
               [ 0],
               [ 1]]), array([[-2, -1,  0,  1]])]
        >>> grid_coord((5, 5))
        [array([[-2],
               [-1],
               [ 0],
               [ 1],
               [ 2]]), array([[-2, -1,  0,  1,  2]])]
        >>> grid_coord((2, 2))
        [array([[-1],
               [ 0]]), array([[-1,  0]])]
        >>> grid_coord((2, 2), dense=True)
        array([[[-1, -1],
                [ 0,  0]],
        <BLANKLINE>
               [[-1,  0],
                [-1,  0]]])
        >>> grid_coord((2, 3), position=(0.0, 0.5))
        [array([[0],
               [1]]), array([[-1,  0,  1]])]
        >>> grid_coord((3, 9), position=(1, 4), is_relative=False)
        [array([[-1],
               [ 0],
               [ 1]]), array([[-4, -3, -2, -1,  0,  1,  2,  3,  4]])]
        >>> grid_coord((3, 9), position=0.2, is_relative=True)
        [array([[0],
               [1],
               [2]]), array([[-1,  0,  1,  2,  3,  4,  5,  6,  7]])]
        >>> grid_coord((4, 4), use_int=False)
        [array([[-1.5],
               [-0.5],
               [ 0.5],
               [ 1.5]]), array([[-1.5, -0.5,  0.5,  1.5]])]
        >>> grid_coord((5, 5), use_int=False)
        [array([[-2.],
               [-1.],
               [ 0.],
               [ 1.],
               [ 2.]]), array([[-2., -1.,  0.,  1.,  2.]])]
        >>> grid_coord((2, 3), position=(0.0, 0.0), use_int=False)
        [array([[ 0.],
               [ 1.]]), array([[ 0.,  1.,  2.]])]
    """
    position = coord(shape, position, is_relative, use_int)
    grid = [slice(-x0, dim - x0) for x0, dim in zip(position, shape)]
    return np.ogrid[grid] if not dense else np.mgrid[grid]


# ======================================================================
def coord(shape, position=0.5, is_relative=True, use_int=True):
    """
    Calculate the coordinate in a given shape for a specified position.

    Args:
        shape (iterable[int]): The shape of the mask in px.
        position (float|iterable[float]): Relative position of the origin.
            Values are in the [0, 1] interval.
        is_relative (bool): Interpret origin as relative.
        use_int (bool): Force interger values for the coordinates.

    Returns:
        position (list): The coordinate in the shape.

    Examples:
        >>> coord((5, 5))
        (2, 2)
        >>> coord((4, 4))
        (2, 2)
        >>> coord((5, 5), 3, False)
        (3, 3)
    """
    position = auto_repeat(position, len(shape), check=True)
    if is_relative:
        if use_int:
            position = tuple(int(scale(x, (0, dim))) for x, dim in zip(position, shape))
        else:
            position = tuple(scale(x, (0, dim - 1)) for x, dim in zip(position, shape))
    elif any([not isinstance(x, int) for x in position]) and use_int:
        raise TypeError("Absolute origin must be integer.")
    return position


# ======================================================================
def scale(val, out_interval=None, in_interval=None):
    """
    Linear convert the value from input interval to output interval

    Args:
        val (float|np.ndarray): Value(s) to convert.
        out_interval (float,float): Interval of the output value(s).
            If None, set to: (0, 1).
        in_interval (float,float): Interval of the input value(s).
            If None, and val is iterable, it is calculated as:
            (min(val), max(val)), otherwise set to: (0, 1).

    Returns:
        val (float|np.ndarray): The converted value(s).

    Examples:
        >>> scale(100, (0, 1000), (0, 100))
        1000.0
        >>> scale(50, (0, 1000), (-100, 100))
        750.0
        >>> scale(50, (0, 10), (0, 1))
        500.0
        >>> scale(0.5, (-10, 10))
        0.0
        >>> scale(np.pi / 3, (0, 180), (0, np.pi))
        60.0
        >>> scale(np.arange(5), (0, 1))
        array([ 0.  ,  0.25,  0.5 ,  0.75,  1.  ])
        >>> scale(np.arange(6), (0, 10))
        array([  0.,   2.,   4.,   6.,   8.,  10.])
        >>> scale(np.arange(6), (0, 10), (0, 2))
        array([  0.,   5.,  10.,  15.,  20.,  25.])
    """
    if in_interval:
        in_min, in_max = sorted(in_interval)
    elif isinstance(val, np.ndarray):
        in_min, in_max = minmax(val)
    else:
        in_min, in_max = (0, 1)
    if out_interval:
        out_min, out_max = sorted(out_interval)
    else:
        out_min, out_max = (0, 1)
    return (val - in_min) / (in_max - in_min) * (out_max - out_min) + out_min


# ======================================================================
def minmax(arr):
    """
    Calculate the minimum and maximum of an array: (min, max).

    Args:
        arr (np.ndarray): The input array.

    Returns:
        min (float): the minimum value of the array
        max (float): the maximum value of the array

    Examples:
        >>> minmax(np.arange(10))
        (0, 9)
    """
    return np.min(arr), np.max(arr)
