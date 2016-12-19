from random import uniform
import numpy as np
import matplotlib.pyplot as plt
from time import time


def source(x_min, x_max, y_min, y_max, theta_min, theta_max, n=1000):

    """
    creates n rays, each characterized by a starting position between x_min and x_max and y_min and y_max,
    by an angular direction between theta_min and theta_max (in degrees).
    The angle theta is the angle between the ray direction and the short side of the rectangular source.
    WARNING: theta_min and max must be given in RADIANS!
    :param x_min, x_max: width of the source in millimeters
    :param y_min, y_max: length of the source in millimeters
    :param theta_min, theta_max: angular span of generated rays wrt the horizontal axis (short side of the source)
    """
    ray_matrix = np.zeros([n, 8])  # 3 columns for initialization, 5 for intersections (4 rings + detector)

    for i in np.arange(n):

        x_0 = uniform(x_min, x_max)
        y_0 = uniform(y_min, y_max)
        theta = uniform(theta_min, theta_max)  # angle between the short side of the rectangular source and the ray

        # I define alpha to be the angle between a point on the collimator ring and the horizontal line
        # (short side of the rectangular source). Thus a point can be described by the radius of its ring and alpha.

        ray_matrix[(i, 0)] = x_0
        ray_matrix[(i, 1)] = y_0
        ray_matrix[(i, 2)] = theta

    return ray_matrix


def collimator_intercepts(r_1, r_2, r_3, r_4, detector_distance, ray_matrix):

    """
    finds the interceptions between the rings and the rays propagating in the collimator
    :param r_1: radius of the first ring measured from the centre of the source
    :param r_2: radius of the second ring measured from the centre of the source
    :param r_3: radius of the third ring measured from the centre of the source
    :param r_4: radius of the fourth ring measured from the centre of the source
    :param detector_distance: detector distance from the centre of the source
    :param ray_matrix: from this matrix the starting point and the direction of the rays are read.
                       the interception points, once calculated, are also written in this matrix
    :return: ray_matrix with the new entries
    """
    n = ray_matrix.shape[0]
    j = 3

    for r in [r_1, r_2, r_3, r_4, detector_distance]:  # checking every "ring"

        for i in np.arange(n):  # checking every ray

            # I'm considering the rays' trajectories as parametric straight lines with time parameter t

            t = - np.cos(ray_matrix[(i, 2)]) * ray_matrix[(i, 0)] - np.sin(ray_matrix[(i, 2)]) * ray_matrix[(i, 1)] \
                + np.sqrt((np.cos(ray_matrix[(i, 2)]) * ray_matrix[(i, 0)] + np.sin(ray_matrix[(i, 2)])
                           * ray_matrix[(i, 1)]) ** 2 + (r ** 2 - ray_matrix[(i, 0)] ** 2 - ray_matrix[(i, 1)] ** 2))

            # point of ray-ring intersection -> fill the matrix up
            # Alpha takes up values between 0 and pi.

            x_intersection = ray_matrix[(i, 0)] + np.cos(ray_matrix[(i, 2)]) * t
            y_intersection = ray_matrix[(i, 1)] + np.sin(ray_matrix[(i, 2)]) * t

            if x_intersection > 0:
                alpha = np.arctan(y_intersection / x_intersection)

            else:
                alpha = np.pi - np.arctan(y_intersection / (-x_intersection))

            if j < 7:
                ray_matrix[(i, j)] = r * alpha  # intersection with first ring

            elif j == 7:
                ray_matrix[(i, 7)] = ray_matrix[(i, 0)] + (r - ray_matrix[(i, 1)]) / np.tan(ray_matrix[(i, 2)])

        j += 1

    return ray_matrix


def bin_generator(r_1, r_2, r_3, r_4, aperture_angle):

    bin_edge = r_1 * 0.5 * (np.pi - aperture_angle)
    a_1 = [bin_edge]
    flag = True

    while bin_edge < r_1 * 0.5 * (np.pi + aperture_angle):  # first ring

        if flag:
            bin_edge += 0.30000

        if not flag:
            bin_edge += 0.34500

        a_1.append(bin_edge)
        flag = not flag

    bin_edge = r_2 * 0.5 * (np.pi - aperture_angle) - 0.0250
    a_2 = [bin_edge]
    flag = False

    while bin_edge < r_2 * 0.5 * (np.pi + aperture_angle):  # second ring

        if flag:
            bin_edge += 0.33396

        if not flag:
            bin_edge += 0.38406

        a_2.append(bin_edge)
        flag = not flag

    bin_edge = r_3 * 0.5 * (np.pi - aperture_angle)
    a_3 = [bin_edge]
    flag = True

    while bin_edge < r_3 * 0.5 * (np.pi + aperture_angle):  # third ring

        if flag:
            bin_edge += 1.1898

        if not flag:
            bin_edge += 1.3683

        a_3.append(bin_edge)
        flag = not flag

    bin_edge = r_4 * 0.5 * (np.pi - aperture_angle) - 0.09196
    a_4 = [bin_edge]
    flag = False

    while bin_edge < r_4 * 0.5 * (np.pi + aperture_angle):  # fourth ring

        if flag:
            bin_edge += 1.2284

        if not flag:
            bin_edge += 1.4127

        a_4.append(bin_edge)
        flag = not flag

    return a_1, a_2, a_3, a_4


def collimator_filtering(a_1, a_2, a_3, a_4, ray_matrix):

    """
    filters out the rays that do not pass through the slits and returns the hits on the detector for the other rays.
    :param a_1, a_2, a_3, a_4:  numpy arrays specifying the positions of the slits for each one of the rings
    :param ray_matrix: matrix with ray starting position (0-1), direction (2), intercepts (3-7)
    :return: detector_hits (array of positions where the rays hit the detector)
    """

    detector_hits = []  # here I store the positions on the detector of the rays that pass through the filter
    n = ray_matrix.shape[0]

    intercepts_1 = ray_matrix[:, 3].flatten()  # arc length (counterclockwise)
    intercepts_2 = ray_matrix[:, 4].flatten()
    intercepts_3 = ray_matrix[:, 5].flatten()
    intercepts_4 = ray_matrix[:, 6].flatten()
    intercepts_detector = ray_matrix[:, 7].flatten()

    bin_number_1 = np.digitize(intercepts_1, a_1)  # array of the same dimension as intercepts
    bin_number_2 = np.digitize(intercepts_2, a_2)
    bin_number_3 = np.digitize(intercepts_3, a_3)
    bin_number_4 = np.digitize(intercepts_4, a_4)

    for i in np.arange(n):  # the rays in bins of odd index can pass

        if bin_number_1[i] % 2 == 0 and bin_number_2[i] % 2 == 1 \
                and bin_number_3[i] % 2 == 0 and bin_number_4[i] % 2 == 1:

                detector_hits.append(intercepts_detector[i])

    return detector_hits


def collimator_filtering2(a_1, a_2, a_3, a_4, ray_matrix):

    """
    filters out the rays that do not pass through the slits and returns the hits on the detector for the other rays.
    :param a_1, a_2, a_3, a_4:  numpy arrays specifying the positions of the slits for each one of the rings
    :param ray_matrix: matrix with ray starting position (0-1), direction (2), intercepts (3-7)
    :return: detector_hits (array of positions where the rays hit the detector)
    """

    detector_hits = []  # here I store the positions on the detector of the rays that pass through the filter
    n = ray_matrix.shape[0]

    intercepts_1 = ray_matrix[:, 3].flatten()  # arc length (counterclockwise)
    intercepts_2 = ray_matrix[:, 4].flatten()
    intercepts_3 = ray_matrix[:, 5].flatten()
    intercepts_4 = ray_matrix[:, 6].flatten()
    intercepts_detector = ray_matrix[:, 7].flatten()

    bin_number_1 = np.digitize(intercepts_1, a_1)  # array of the same dimension as intercepts
    bin_number_2 = np.digitize(intercepts_2, a_2)
    bin_number_3 = np.digitize(intercepts_3, a_3)
    bin_number_4 = np.digitize(intercepts_4, a_4)

    # for i in np.arange(n):  # the rays in bins of odd index can pass
    #
    #     if bin_number_1[i] % 2 == 0 and bin_number_2[i] % 2 == 1 \
    #             and bin_number_3[i] % 2 == 0 and bin_number_4[i] % 2 == 1:
    #
    #             detector_hits.append(intercepts_detector[i])

    # print("min=%f, max=%f"%(bin_number_1.min(), bin_number_2.max()))

    flag = np.ones_like(intercepts_1)

    for i in range(0,bin_number_1.max()+1):

        if i % 2 == 0:
            flag[np.where(bin_number_2 == i)] = 0.0
            flag[np.where(bin_number_4 == i)] = 0.0
        else:
            flag[np.where(bin_number_1 == i)] = 0.0
            flag[np.where(bin_number_3 == i)] = 0.0

    detector_hits = intercepts_detector[np.where(flag == 1.0)]

    return detector_hits


if __name__ == "__main__":

    # get some plotting parameters from user

    ray_num = int(input("How many rays in the simulation? (pick a multiple of 1000) "))
    bins = int(input("How many bins in the histogram? "))
    loop_num = ray_num // 1000  # how many loops do I have to run to get to the desired number of rays?

    # print an estimate of the calculation time

    time_file = open("time_record.txt", "r")
    total = 0.0
    line_counter = 0.0

    for line in time_file:

        total += float(line)
        line_counter += 1.0

    time_file.close()

    time_estimate = total / line_counter * loop_num
    print("This calculation is expected to take {0} minutes {1} seconds".format(time_estimate//60, round(time_estimate%60, 2)))

    # collimator parameters

    aperture_angle = 60 * np.pi / 180  # 45 degrees aperture
    r_1 = 53.000  # millimeters
    r_2 = 59.000
    r_3 = 210.20
    r_4 = 217.02
    detector_distance = 500.00  # WARNING: NOT THE ACTUAL VALUE

    # source parameters

    x_min = -1.5e-3  # millimeters
    x_max = 1.5e-3
    y_min = -2.0  # millimeters
    y_max = 2.0

    # theta min and max are calculated so that the whole collimator is illuminated

    theta_min = np.pi / 2 - aperture_angle / 2 \
        - np.arctan((y_max * np.sin(aperture_angle / 2)) / (r_1 - y_max * np.cos(aperture_angle / 2)))
    theta_max = np.pi - theta_min

    # build the binning sequences

    a_1, a_2, a_3, a_4 = bin_generator(r_1, r_2, r_3, r_4, aperture_angle)
    a_1 = np.asarray(a_1)
    a_2 = np.asarray(a_2)
    a_3 = np.asarray(a_3)
    a_4 = np.asarray(a_4)

    # >>> if you want to plot the slits you can copy "draw_slits_script.txt" here <<<

    # split the calculation up into 1000-ray subroutines
    detector_hits = []
    time_record = []

    for i in np.arange(loop_num):

        start_time = time()  # get time since the epoch in seconds

        ray_matrix = source(x_min, x_max, y_min, y_max, theta_min, theta_max, 1000)

        ray_matrix = collimator_intercepts(r_1, r_2, r_3, r_4, detector_distance, ray_matrix)

        detector_hits = np.concatenate([detector_hits, collimator_filtering(a_1, a_2, a_3, a_4, ray_matrix)])

        stop_time = time()

        time_record.append(stop_time - start_time)

    time_average = np.average(time_record)
    print("average time per loop: {} seconds".format(time_average))
    time_sum = np.sum(time_record)
    print("total elapsed time: {0} minutes {1} seconds".format(time_sum // 60, round(time_sum % 60, 2)))

    # keep a time record

    time_file = open("time_record.txt", "a")
    time_file.write("\n" + str(time_average))
    time_file.close()

    # save data

    np.save("data60.npy", detector_hits)

    # plot the results as a histogram

    plt.figure()
    plt.title("detector counts")
    plt.xlabel("distance from the central axis")
    plt.ylabel("counts")
    plt.hist(detector_hits, bins=bins)
    plt.show()
