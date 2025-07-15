# This test is based on the toy example notebook
import os
from pathlib import Path
from kcpdi.kcp_ds import kcp_ds
import pickle

ROOT_PATH = str(Path(__file__).parent.resolve()) + os.sep # To point on test directory

def test_kcpdi():

    # Save to pickle file
    with open(ROOT_PATH+'test-ds.pkl', 'rb') as f:
        my_data=pickle.load(f)

    detected_change_points, interval_end_points = kcp_ds(my_data, expected_frac_anomaly=15/1000)
    expected_change_point=[113, 213, 291, 336, 412, 520, 667, 807, 930]
    # print('The detected change-points are at times: ', detected_change_points)
    assert detected_change_points==expected_change_point
    