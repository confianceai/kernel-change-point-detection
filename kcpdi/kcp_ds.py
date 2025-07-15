"""
This module, `kcp_ds` (Kernel Change Point Detect Select), encapsulates functions from the Ruptures
package and includes a function for penalized variable selection.

Specific details and remarks:
1. The function outputs Python indices of the columns of "data" corresponding to the detected
change points. Real-world times associated with the data are not considered in this function.
2. The ruptures class KernelCPD has a minimum allowed segment size (min_size). It is currently set
to 2 by default and is not an input parameter for kcp_ds.
3. The default kernel is "linear," with options for "cosine" and "rbf" kernels. To cover all bases,
an input params is required, which defaults to None but can be modified for the rbf kernel (e.g.,
params = {"gamma": value_of_gamma}).
4. Due to memory constraints, the algorithm processes offline time-series data longer than max_n_time_points
in segments of length max_n_time_points, and then outputs the full set of change points.

Note: Please handle real-world times associated with the data independently as this module focuses
on change point detection and variable selection.
"""


from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple
import math

import numpy as np
import ruptures as rpt
from sklearn.linear_model import LinearRegression

from kcpdi.utils import get_sum_of_cost


def kcp_ds(
    data: np.array,
    kernel: Literal["linear", "cosine", "rbf"] = "linear",
    params: Optional[Dict[str, Any]] = None,
    max_n_time_points: int = 2000,
    min_n_time_points: int = 10,
    expected_frac_anomaly: float = 1/1000,
) -> Tuple[List[int], List[int]]:
    """Return a set of important change points change points after
    running kernel change point detection followed by penalized model
    selection using the slope heuristic method.

    Since the criterion is responding to noise/overfitting, penalized
    variable selection is performed in order to obtain a final decision
    as to how many of the detected change points are truly significant
    and not simply adjusted noise. 

    Args:
        data: array of dimension (number of time points) x (number of
            time-series) containing the data points.
        kernel: Kernel of the kernet change_point detection. If it is
            rbf, params must include a "gamma" parameters, with a
            positive real value.
        params: parameters for the kernel instance
        max_n_time_points: maximum size (max_n_time_points) x
            (max_n_time_points) of matrices expected to be processed
            quickly by the computer system.
        min_n_time_points: minimum number of time points in the current
            dataset for which it makes sense to run the detection
            algorithm again. If there are fewer than min_n_time_points
            points in the dataset, no computations will be run and the
            outputs will be empty.
        expected_frac_anomaly: This parameter encodes prior knowledge 
            of users as to how often anomalies might occur. Results can
            be quite dependent on the choice of this parameter, so 
            choose carefully!
    """
    if max_n_time_points < min_n_time_points:
        raise ValueError(
            "max_n_time_points should be greater than min_n_time_points."
        )
        
    if expected_frac_anomaly > 1/2:
        raise ValueError(
            "expected_frac_anomaly should be at most 1/2."
        )

    n_total_time_points = np.shape(data)[0]

    current_data = np.copy(data)  # new variable for internal manipulation

    n_current_time_points = np.shape(current_data)[0]
    curr_n_remaining_t_points = n_current_time_points

    # Initialise an empty list to store the end points of the intervals of length max_n_time_points
    # (except for the last interval, which could be shorter). This variable will be required for
    # the explainability function that can be run after the current function:
    interval_end_points = []

    current_zero = 0  # remember the original Python indices

    detected_change_points = []

    while curr_n_remaining_t_points > max_n_time_points:
        # Get the block of data to plug into the Ruptures setting:
        data_use = current_data[
            :max_n_time_points,
            :
        ]

        # Store the true index of the time point at the end of this interval, which will be needed
        # in the explainability function later:
        interval_end_points.append(current_zero + max_n_time_points)

        detected_change_points, next_ch_pts = _detect_changepoints_from_kernelcpd(
            kernel=kernel,
            params=params,
            data_use=data_use,
            expected_frac_anomaly=expected_frac_anomaly,
            current_zero=current_zero,
            detected_change_points=detected_change_points,
        )

        # Concatenate any current data rows left after the last change point found, to the
        # remaining data (if there is any left):
        current_data = data[(next_ch_pts[-1]-1):, :]

        current_zero = next_ch_pts[-1] - 1

        n_current_time_points = np.shape(current_data)[0]
        curr_n_remaining_t_points = n_current_time_points

    # Deal with any residual block of data that is shorter than (or equal to) 
    #max_n_time_points but still larger than min_n_time_points:
    if (
        curr_n_remaining_t_points <= max_n_time_points
    ) & (
        curr_n_remaining_t_points >= min_n_time_points
    ):

        # Store the true index of the time point at the end of this interval, which will be needed
        # in the explainability function later. Unlike in the above while loop, here the time point
        # at the end of the interval is the last time point in the dataset:
        interval_end_points.append(n_total_time_points)

        detected_change_points, next_ch_pts = _detect_changepoints_from_kernelcpd(
            kernel=kernel,
            params=params,
            data_use=current_data,
            expected_frac_anomaly=expected_frac_anomaly,
            current_zero=current_zero,
            detected_change_points=detected_change_points,
        )
        
    #We need to remove any "detected" change-points which were actually ends of intervals.
    #First we add 1 to each detected_change_point:
    temp_detec = [detected_change_points[i] + 1 for i in range(len(detected_change_points))]
    #Now look for matches with values in "interval_end_points"
    index_matches = [i for i in range(len(detected_change_points)) if temp_detec[i] in interval_end_points]
    detected_change_points = [i for j, i in enumerate(detected_change_points) if j not in index_matches]

    return detected_change_points, interval_end_points


def _detect_changepoints_from_kernelcpd(
    kernel: Literal["linear", "cosine", "rbf"],
    params: Optional[Dict[str, Any]],
    data_use: np.array,
    expected_frac_anomaly: float,
    current_zero: int,
    detected_change_points: List[int],
) -> Tuple[List[int], List[int]]:
    """Apply kernel change point detection and update
    detected_change_points.

    Auxiliary function to kcp_ds. Update the detected_change_points
    variable in place.

    Args:
        kernel: Kernel of the kernet change_point detection. If it is
            rbf, params must include a "gamma" parameters, with a
            positive real value.
        params: parameters for the kernel instance
        data_use: chunk of data on which to apply the detection
        expected_frac_anomaly: This parameter encodes prior knowledge 
            of users as to how often anomalies might occur. Results can
            be quite dependent on the choice of this parameter, so 
            choose carefully!
        current_zero: index of the first point of the current chunk of 
            the dataset
        detected_change_points: all breakpoints detected so far

    Returns:
        The list of all points detected so far.
        The list of points newly detected during the last iteration.
    """
    # Associate the data with the class KernelCPD from the Ruptures library:
    algo = rpt.KernelCPD(kernel=kernel, params=params)
    algo.fit(data_use)
    
    #Get a value for the maximum number of change-points to look for; essentially,
    #this is set as = 2 * fraction of expected anomalies * number of current time points.
    n_bkps_max = max( 1, int(2*np.floor(expected_frac_anomaly*np.shape(data_use)[0])) )
    
    array_of_n_bkps = np.arange(1, n_bkps_max + 1)
    algo_costs = [
        get_sum_of_cost(algo=algo, n_bkps=n_bkps)
        for n_bkps in array_of_n_bkps
    ]  # value of the criterion for when we have chosen to stick with 1 up to n_bkps_max points
    # Since each time we add an extra change point, the criterion will never decrease:
    # algo_costs is a non-increasing list.
    
    #For the linear kernel we want to concatenate onto the FRONT of this list the value of the
    #criterion when there are ZERO change-points. The problem is that the ruptures package 
    #assumes that there is at least one change-point, while the model selection of Arlot et al
    #(2019) includes the case when there are ZERO change-points. We emphasize that the matrix
    #multiplication below is only for the linear kernel.
    linear_prod = np.dot(data_use,data_use.T)
    diag_sum = np.sum([linear_prod[i,i] for i in range(np.shape(data_use)[0])])
    crit_linear_zero_cp = diag_sum - np.mean(linear_prod)
    #Concatenate this value onto the FRONT of the algo_costs list:
    algo_costs.insert(0,crit_linear_zero_cp)

    n_bkps_pen = _model_select_n_bp(
        data=data_use,
        algo_costs=algo_costs,
        array_of_n_bkps=array_of_n_bkps,
        n_bkps_max=n_bkps_max,
    )  # penalized selection
    
    # Extract the solution:
    if n_bkps_pen > 0:
        next_ch_pts = algo.predict(n_bkps=n_bkps_pen)
        # Remove the end point, which itself is not an endpoint:
        del next_ch_pts[-1]
    else: 
        #If there were NO change-points detected, by convention we shall 
        #instead output the end-point index of the interval AS IF it were
        #a change-point. Note that this (index + 1) was also already stored in
        #the variable "interval_end_points" so we will be able to weed out
        #this change-point at the end for the final list and for the
        #eventual visualization steps.
        next_ch_pts = algo.predict(n_bkps=1)
        next_ch_pts = [next_ch_pts[-1]-1]
    
    # Add the current zero index (in order to get the true row/time point in the data):
    next_ch_pts = [x + current_zero for x in next_ch_pts]

    # Concatenate these with any found in previous loops:
    detected_change_points = detected_change_points + list(next_ch_pts)

    return detected_change_points, next_ch_pts


def _model_select_n_bp(
    data: np.array,
    algo_costs: Sequence[float],
    array_of_n_bkps: Sequence[int],
    n_bkps_max: int,
) -> int:
    """Calculate the number of change points after model selection using
    the slope heuristic method.

    This function takes the output set of candidate change points from
    the Ruptures library method and refines them to a subset that is
    likely to represent signal rather than (over)fitted noise. It
    implements the penalized variable selection method described
    in Arlot et al. (2019).

    Args:
        data: signal data points in which to calculate the optimal
            number of change points
        algo_costs: the (non-penalized) costs associated with each
            number of change points
        array_of_n_bkps: array listing the number of change points to be
            considered
        n_bkps_max: maximum number of change points to be detected

    Returns:
        The number of change points after penalized variable selection
        using the slope heuristic method in Arlot et al. (2019).
    """
    lower_D = math.ceil(0.6 * n_bkps_max)
    
    X1 = [
        (1/np.shape(data)[0])*math.log(math.comb(np.shape(data)[0] - 1, D - 1))
        for D in np.arange(lower_D, n_bkps_max + 1)
    ]
    X2 = [(1/np.shape(data)[0])*D for D in np.arange(lower_D, n_bkps_max + 1)]
    
    X = np.transpose(np.array([X1, X2]))
    y = np.array([(1/np.shape(data)[0])*algo_costs[D - 1]
                 for D in np.arange(lower_D, n_bkps_max + 1)])
    #
    ## Linear regression:
    algo_reg = LinearRegression().fit(X, y)
    s1_hat = algo_reg.coef_[0]
    s2_hat = algo_reg.coef_[1]
     
    ###########################################################
    
    # Final parameters:
    c1 = -2 * s1_hat
    c2 = -2 * s2_hat

    # Use these two parameters to do the model selection in Arlot et al. (2019):         
    
    algo_pen = [
        algo_costs[i-1]
        +
        c1 * math.log(math.comb(np.shape(data)[0] - 1, i - 1))
        +
        c2 * i
        for i in array_of_n_bkps
    ]
    
    min_val = min(algo_pen)
    all_min_indices = [i for i, x in enumerate(algo_pen) if x == min_val]

    # There could be more than one items in this object. We choose the smallest for parcimony
    # reasons in this (unlikely) event:
    min_index = all_min_indices[0]
    
    n_bkps_pen = min_index

    return n_bkps_pen
