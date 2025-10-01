"""
This module, `kcp_v` (Kernel Change Point Visualisation), includes a function for visualization and
explainability. It extracts intervals around each detected change point, illustrating the
contribution of each sensor to the change point criterion. Each interval is bounded on the left by
either 0 or the previous change point location and on the right by either the next change point or
the time series end-point. The visualization highlights the change point's location with a vertical
dotted red line and displays the algorithm criterion as a blue curve.

Remarks:
1. For clarity in visualization, individual time series related to different change points are
visualized separately, resulting in one plot per detected change point.
2. The visualization function accommodates the algorithm's processing of a maximum of
"max_n_time_points" in each loop. If the total exceeds "max_n_time_points," the interval
calculation differs for the last change point found in each loop. Hence, "max_n_time_points" must
be consistent between the change point detection and visualization functions.
3. Only the linear kernel provides criterion lines for each individual time series that sum up to
the value of the global criterion line. Although designed for the linear kernel, the visualization
function can be used for other kernels (if the same kernel is used in the kcp_ds function).
4. #data: The input data array must be the exact same array used in the kcp_ds function.
"""

from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt


def kcp_v(
    data: np.array,
    true_times: Sequence[int],
    detected_change_points: Sequence[int],
    interval_end_points: Sequence[int],
    n_legend: int = 10,
    save_plots: bool = False,
) -> None:
    """Visualize the post-hoc importance of each individual time series
    in the detection of each shared change point by the kernel change
    point detection algorithm.

    Args:
        data: array of dimension (number of time points) x (number of
            time-series) containing the data points.
        true_times: the dataset-dependent true time points (after any
            postprocessing steps required to interpolate data to a fixed
            time grid). This is required to make more meaningful plot
            outputs, or to obtain the actual true times of predicted
            change points. true_times should have the length of the
            first dimension of data.
        detected_change_points: the detected change points. It should be
            the first output of the kcp_ds function.
        interval_end_points: indices of the ends of intervals of length
            max_n_time_points. It should be the second output of kcpDS.
        n_legend: the number of "important" individual time series you
            want to be colorful in the final plot.
    """
    # Calculate the number of individual time series in parallel:
    n_time_series = np.shape(data)[1]

    # Stock the times that will be used to plot the criterion value. These times are halfway
    # between the actual data times, since when we say we detect a change point at time time, we
    # actually mean between t and the next data time, so for simplicity, we make this half-way
    # between the two:
    plot_times = [
        (true_times[i] + true_times[i + 1]) / 2 for i in range(len(true_times) - 1)
    ]

    # For each change point in detected_change_points:
    for i, _ in enumerate(detected_change_points):
        print("Change point number", i + 1, "\n")

        # Left edge Python index:
        L = -1 * (i == 0) + (detected_change_points[i - 1] - 1) * (i > 0)

        # Right edge Python index:
        # (Various cases to deal with here: (i) It's the last change point detected,
        # (ii) It's the last change point detected in a time block of length
        # "max_n_time_points", in which case the right edge is the end of the time
        # block, not the next change point)
        # (iii) Neither of the above:

        # If it is not the last change point detected:
        if i < len(detected_change_points) - 1:
            # If it is not the last change point in a block of "max_n_time_points"
            # time points: (to do this we need to look at the list "interval_end_points"
            # and check whether the (i+1)th and (i+2)th detected change points
            # are both smaller than the same number of elements in the list, i.e., whether
            # the were both found in the same loop of in kcpDS, or not)
            if np.sum(
                detected_change_points[i] < np.array(interval_end_points)
            ) == np.sum(detected_change_points[i + 1] < np.array(interval_end_points)):
                R = detected_change_points[i + 1] - 1
            else:
                # Extract the end point of the current interval from "interval_end_points", since
                # it comes BEFORE the next actually detected change point.
                n_greater_than = np.sum(
                    detected_change_points[i] > np.array(interval_end_points)
                )
                R = interval_end_points[int(n_greater_than)] - 1

        # If it is the last change point detected:
        else:
            R = np.shape(data)[0] - 1

        # Central change point:
        C = detected_change_points[i] - 1

        # Here we pull out the sub-dataset from times L to R, in order to examine it closer
        # and hopefully get some understanding on why, given this data, the change point was
        # detected at C.

        # arlot_crit is a list that will store the value of the kernel change point criterion for
        # each possible time at which we could have placed a change point between L and R. Due do
        # the fact that restricted to this interval, the solution for finding one (and one only)
        # change point will give the same change point as found by kcpDS globally (technical detail
        # due to how the criterion is calculated), we expect the MINIMUM of arlot_crit to be at C.
        # If it is not, then most likely Python indexing starting at 0 has probably created another
        # minor error somewhere. Thanks Python!
        arlot_crit = [0] * (R - L - 1)

        # Put our current data into an array:
        LR_data = np.transpose(data[(L + 1): (R + 1), :])

        # First calculate the diagonal term for the criterion for the LINEAR KERNEL:
        arlot_crit_diag = (1 / (R - L)) * np.sum(
            [np.sum(np.multiply(LR_data[:, j], LR_data[:, j])) for j in range(0, R - L)]
        )

        # Also, for later, calculate the criterion as if there were NO change points between L and R:
        arlot_crit_zero = arlot_crit_diag - (1 / (R - L)) * (1 / (R - L)) * np.sum(
            np.matmul(np.transpose(LR_data), LR_data)
        )

        # Now we cycle through the possible places to put one change point between L and R. This for
        # loop can be slow. There is no reason why it can't be parallelized if you are good enough at
        # Python to do so.
        for z in range(0, R - L - 1):
            C_z = L + 1 + z
            L_data_z = np.transpose(data[(L + 1): (C_z + 1), :])
            R_data_z = np.transpose(data[(C_z + 1): (R + 1), :])
            arlot_crit[z] = arlot_crit_diag - (1 / (R - L)) * (
                (1 / (C_z - L)) * (np.sum(np.matmul(np.transpose(L_data_z), L_data_z)))
                + (1 / (R - C_z))
                * (np.sum(np.matmul(np.transpose(R_data_z), R_data_z)))
            )

        # Now we must do exactly the same for each individual time series. Basically it is the
        # same code as just above, but for each time series, one at a time.

        # arlot_crit_sense_array has the same job as arlot_crit above, except it requires
        # one row per time series:
        arlot_crit_sense_array = np.zeros((n_time_series, R - L - 1))

        # arlot_crit_zero_zz has the same function as arlot_crit_zero above.
        arlot_crit_zero_zz = [0] * n_time_series

        # Here we cycle through the individual time series, one by one. There is no
        # barrier to parallelizing this.
        for m in range(0, n_time_series):
            arlot_crit_sense = [0] * (R - L - 1)
            LR_data_zz = np.transpose(data[(L + 1): (R + 1), m]).reshape(1, -1)

            # First calculate the diagonal:
            arlot_crit_diag_zz = (1 / (R - L)) * np.sum(
                [
                    np.sum(np.multiply(LR_data_zz[0, j], LR_data_zz[0, j]))
                    for j in range(0, R - L)
                ]
            )

            # Also calculate the criterion as if there were no change points, for later use.
            arlot_crit_zero_zz[m] = arlot_crit_diag_zz - (1 / (R - L)) * (
                1 / (R - L)
            ) * (np.sum(np.matmul(np.transpose(LR_data_zz), LR_data_zz)))

            # This inner for loop could also be parallelized.
            for z in range(0, R - L - 1):
                C_z = L + 1 + z
                L_data_zz = np.transpose(data[(L + 1): (C_z + 1), m]).reshape(1, -1)
                R_data_zz = np.transpose(data[(C_z + 1): (R + 1), m]).reshape(1, -1)
                arlot_crit_sense[z] = arlot_crit_diag_zz - (1 / (R - L)) * (
                    (1 / (C_z - L))
                    * (np.sum(np.matmul(np.transpose(L_data_zz), L_data_zz)))
                    + (1 / (R - C_z))
                    * (np.sum(np.matmul(np.transpose(R_data_zz), R_data_zz)))
                )

            arlot_crit_sense_array[m, :] = arlot_crit_sense

        plt.figure(figsize=(15, 10))

        plot_points = plot_times[(L + 1): R]

        plt.plot(plot_points, (arlot_crit - arlot_crit_zero), label="Combined")

        plt.axvline(plot_times[(detected_change_points[i] - 1)], ls="--", c="red", lw=1)

        plt.xlabel("Time")
        plt.ylabel("Kernel change-point criterion")

        # Here we want to look at the value of the criterion for each individual time series
        # at C, and rank them from smallest to largest. We will then plot the first few smallest
        # on the same plot as the global criterion.
        val_at_global_min = [
            arlot_crit_sense_array[x, C - L - 1] - arlot_crit_zero_zz[x]
            for x in range(n_time_series)
        ]

        sorted_indices = np.argsort(val_at_global_min)

        # The ones we keep
        keep_indices = sorted_indices[0:n_legend]

        for m in range(0, n_time_series):
            if m in keep_indices:
                plt.plot(
                    plot_times[(L + 1): R],
                    arlot_crit_sense_array[m, :] - arlot_crit_zero_zz[m],
                    label=m,
                )
            else:
                plt.plot(
                    plot_times[(L + 1): R],
                    arlot_crit_sense_array[m, :] - arlot_crit_zero_zz[m],
                    color="black",
                )
        plt.legend(title="Important time series")

        if save_plots:
            plt.savefig(f"plot{i}")

        plt.show()
