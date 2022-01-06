# This file provides functions for analysis of the quality of reconstructed images

# Erik Riedel & Josef Kamysek
# erik.riedel@tum.de & josef@kamysek.com
# 5/11/2020, Munich, Germany

# Python imports
import math
import logging

# Third party libraries
import numpy as np
from matplotlib import pyplot as plt

logger = logging.getLogger("Waveblocks")


def lin_interp(data, i, half):
    return i + ((half - data[i]) / (data[i + 1] - data[i]))

def fwhm_analysis(volume, coordinates, distances, dimension=1):
    """
    To use with reconstructed microsphere volume. This function will evalueate the average full width at half maximum of each of the coordinates in Z direction

    volume: reconstructed volume containing microspheres
    coordinates: coordinates of microsphere center
    distances: distances between microsphere centers
    dimension: In which direction to examine the FWHM (x=2, y=3, z=1)
    """
    hmz = []

    # normalize volume
    min_val = np.min(volume)
    volume = volume - min_val
    max_val = np.max(volume)
    volume = volume / max_val

    num_coords = len(coordinates)
    valid_coords = 0

    for (x, y, z) in coordinates:

        # select relevant 1 dimensional sample
        if dimension == 1:
            data = volume[0, :, x, y]
        elif dimension == 2:
            data = volume[0, z, :, y]
        elif dimension == 3:
            data = volume[0, z, x, :]
        else:
            raise RuntimeError("Invalid Dimension given for FWHM")

        # crop relevant data
        start = max(0, math.floor((z - (distances / 2))))
        end = min(data.shape[0], math.ceil(z + (distances / 2)))
        data = data[start : end + 1]

        half = 0.5

        # plt.plot(range(len(data)), data)
        # plt.plot(range(len(data)), np.repeat(half,len(data)))
        # plt.show()

        signs = np.sign(np.add(data, -half))
        zero_crossings = signs[0:-1] != signs[1:]
        zero_crossings_i = np.where(zero_crossings)[0]

        if max(data) < half or min(data) > half:
            logger.debug("All function values either higher or lower than half maximum")
            hmz.append((start, end, 0))
            continue

        if len(zero_crossings_i) != 2:
            logger.debug(
                "Can not accurately compute fwhm as there are multiple spikes greater than half-maximum"
            )
            hmz.append((start, end, 0))
            continue

        valid_coords += 1

        hmz.append(
            (
                start + lin_interp(data, zero_crossings_i[0], half),
                start + lin_interp(data, zero_crossings_i[1], half),
                half,
            )
        )

    logger.info(
        str(valid_coords)
        + "/"
        + str(num_coords)
        + " coordinates have a valid function for FWHM calculation"
    )

    show_first_fwhm = True
    if show_first_fwhm:

        test_coord = coordinates[0]
        
        if dimension == 1:
            test_row = volume[0, :, test_coord[0], test_coord[1]]
        elif dimension == 2:
            test_row = volume[0, test_coord[2], :, test_coord[1]]
        else:
            test_row = volume[0, test_coord[2], test_coord[0], : ]


        # a convincing plot
        half = max(test_row) / 2.0

        logger.info("coordinates of first sphere: {}".format(test_coord))

       
        plt.title("FWHM for first sphere row")
        plt.plot(hmz[0][0:2], [hmz[0][2], hmz[0][2]])
        plt.plot(range(test_row.shape[0]), test_row)
        plt.show()

    return hmz