# Python imports
from enum import Enum

# Third party libraries
import numpy as np

# Waveblocks imports
import waveblocks.evaluation.microspheres as wbm


class ObjectType(Enum):
    emerald = 1
    sphere = 2


def create_emerald(radius):
    """
    Generates a emerald given a certain radius.

    Args:
        radius: The radius of the emerald in px.

    Returns:
        mask: Array of float describing the emerald.
    """

    # cx, cy, cz: center coordinates.
    # size: size 3D matrix.
    cx, cy, cz = radius, radius, radius
    size = 2 * radius + 1

    # matrix: 3D matrix of shape size * size * size.
    matrix = np.zeros((size, size, size))

    for x in range(cx - radius, cx + radius + 1):
        for y in range(cy - radius, cy + radius + 1):
            for z in range(cz - radius, cz + radius + 1):
                """
                deb: measures how far a coordinate in A is far from the center.
                deb>=0: inside the sphere.
                deb<0: outside the sphere.
                """
                deb = radius - abs(cx - x) - abs(cy - y) - abs(cz - z)
                if deb >= 0:
                    matrix[x, y, z] = 1

    return matrix


def create_sphere(radius):
    """
    Generates a sphere given a certain radius.

    Args:
        radius: The radius of the sphere in px.

    Returns:
        mask: Array of float describing the sphere.
    """

    shape = 2 * radius + 1   

    return wbm.sphere(shape, radius).astype(float)


def create_volume(radius, vx, vy, vz, type, distance=0, coordinates=[], zero_space=0):
    """
    Generates a volume filled with the specified geometric object.

    Args:
        radius: The radius of the emerald/sphere in px.
        vx, vy, vz: Size of the volume in x, y, z direction.
        type: Describes the object type.
        distance: Specifies the distance between the periodically create objects. (objects) <--> (objects)
        coordinates: Allows to explicitly specify the coordinates where the objects should be placed.

    Returns:
        volume: Volume containing the objects
        volume_coordinates: Array containing the coordinates of each object
        :param vx:
        :param vy:
        :param vz:
        :param zero_space:
    """

    # Create object for defined type
    object = None
    if not isinstance(type, ObjectType):
        raise TypeError("Type must be an instance of ObjectType")

    if type == ObjectType.emerald:
        object = create_emerald(radius)

    if type == ObjectType.sphere:
        object = create_sphere(radius)

    volume = np.zeros((vx, vy, vz))

    vx = vx - 2 * zero_space
    vy = vy - 2 * zero_space
    vz = vz - 2 * zero_space

    # Use provided coordinates or create
    if len(coordinates) == 0:
        distance = radius + 1 + distance
        
        # Check if shape of object is smaller than shape of volume
        if (
            not object.shape[0] <= volume.shape[0]
            or not object.shape[1] <= volume.shape[1]
            or not object.shape[2] <= volume.shape[2]
        ):
            raise Exception("shape of object is bigger than volume")

        # Cycle through volume and place objects and store center coordiantes of objects
        x = 0 + zero_space
        while x + object.shape[0] < vx + zero_space:
            y = 0 + zero_space
            while y + object.shape[1] < vy + zero_space:
                z = 0 + zero_space
                while z + object.shape[2] < vz + zero_space:
                    # Update volume and store coordiates
                    volume[
                        x : x + object.shape[0],
                        y : y + object.shape[1],
                        z : z + object.shape[2],
                    ] = object[:, :, :]
                    coordinates.append((x + radius, y + radius, z + radius))
                    z = z + (object.shape[2] + distance - 2 * radius)

                y = y + (object.shape[1] + distance - 2 * radius)

            x = x + (object.shape[0] + distance - 2 * radius)

    else:
        for sc in coordinates:
            # Check if object is out of volume
            if (
                not sc[0] <= volume.shape[0]
                or not sc[1] <= volume.shape[1]
                or not sc[2] <= volume.shape[2]
            ):
                raise Exception("shape of object is bigger than volume")

            # Update volume
            volume[
                sc[0] - radius : sc[0] - radius + object.shape[0],
                sc[1] - radius : sc[1] - radius + object.shape[1],
                sc[2] - radius : sc[2] - radius + object.shape[2],
            ] = object[:, :, :]

    # Returns volume with objects and a list of all the middle points of the objects.
    return volume, coordinates


"""
Test Sphere Implementation

sphere_volume, sphere_coordinates = create_sphere_volume(10, 50, 50, 50)
print(sphere_volume.shape)
print(sphere_volume)
print(sphere_coordinates)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x, y, z = sphere_volume.nonzero()
ax.scatter(x , y , z)
plt.show()

sphere_volume, sphere_coordinates = create_sphere_volume(10, 50, 50, 50, 0,[(25,25,25)])
print(sphere_volume.shape)
print(sphere_volume)
print(sphere_coordinates)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x, y, z = sphere_volume.nonzero()
ax.scatter(x , y , z)
plt.show()

sphere_volume, sphere_coordinates = create_sphere_volume(3, 50, 50, 50, sphere_coordinates=[(4,4,4)])
print(sphere_volume.shape)
print(sphere_volume)
print(sphere_coordinates)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x, y, z = sphere_volume.nonzero()
ax.scatter(x , y , z)
plt.show()

"""
