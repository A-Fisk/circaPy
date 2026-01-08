import unittest
import sys
import os
import pdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
if True:  # noqa E402
    from tests.activity_tests import assign_values, generate_test_data
    from circaPy.preprocessing import set_circadian_time, validate_input


class TestSetCircadianTime(unittest.TestCase):

    def setUp(self):
        """Set up sample data for testing using the provided functions"""
        # Generate test data with 10 days of data at 10-second intervals
        self.data = generate_test_data(
            days=10,
            freq="10s",
            act_night=[0, 10],
            act_day=[10, 100],
            light_night=[0, 1],
            light_day=[500, 501],
        )

    def test_set_circadian_time_with_str_period(self):
        """Test if the function handles string period (e.g., '24h') correctly"""
        period = "24h"
        result = set_circadian_time(self.data, period)

        # Ensure the result has the same number of rows as the original data
        self.assertEqual(len(result), len(self.data))

        # Check the new frequency (should be approximately hourly)
        self.assertTrue(result.index.freqstr in ["10000ms"])

    def test_set_circadian_time_with_other_timedelta(self):
        """Test if the function handles other timedelta periods (e.g., '72h') correctly"""
        period = "72h"
        result = set_circadian_time(self.data, period)

        # Ensure the result has the same number of rows as the original data
        self.assertEqual(len(result), len(self.data))

        # Check the new frequency (should be 3333ms for '72h' period)
        self.assertTrue(result.index.freqstr in ["3333ms"])

    def test_set_circadian_time_with_nonstandard_period(self):
        """Test if the function handles non-standard periods (e.g., '1d') correctly"""
        period = "1d"  # equivalent to 24 hours
        result = set_circadian_time(self.data, period)

        # Ensure the result has the same number of rows as the original data
        self.assertEqual(len(result), len(self.data))

        # Check the new frequency (should be approximately 10000ms for '1d')
        self.assertTrue(result.index.freqstr in ["10000ms"])

    def test_set_circadian_time_with_invalid_period(self):
        """Test if the function raises an error for invalid period inputs"""
        period = "invalid_period"

        # Check if ValueError is raised for invalid period string
        with self.assertRaises(ValueError):
            set_circadian_time(self.data, period)

    def test_set_circadian_time_with_no_period(self):
        """Test if the function uses default period when none is provided"""
        result = set_circadian_time(self.data)

        # Ensure the result has the same number of rows as the original data
        self.assertEqual(len(result), len(self.data))

        # Check the default period is '24h'
        self.assertTrue(result.index.freqstr in ["10000ms"])

    def test_set_circadian_time_frequency_preservation(self):
        """Test if the frequency of the resulting data is correctly adjusted"""
        period = "48h"  # 48 hours as the new period
        result = set_circadian_time(self.data, period)

        # Ensure the new frequency matches the expected adjustment
        # 5000ms should be the new frequency
        self.assertTrue(result.index.freqstr in ["5000ms"])

    def test_set_circadian_time_with_alternate_freq(self):
        """Test if the function handles non-evenly sampled data correctly"""
        # Generate data at a 1-minute frequency (uneven sampling)
        data_uneven = generate_test_data(
            days=5,
            freq="1min",
            act_night=[0, 10],
            act_day=[10, 100],
            light_night=[0, 1],
            light_day=[500, 501],
        )

        period = "48h"
        result = set_circadian_time(data_uneven, period)

        # Ensure the result has the same number of rows as the original data
        self.assertEqual(len(result), len(data_uneven))

        # Check the new frequency (should be 30000ms intervals for '48h'
        # period)
        self.assertTrue(result.index.freqstr in ["30000ms"])


# Test validate input decorator
class TestValidateInput(unittest.TestCase):

    def setUp(self):
        """Set up sample data for testing using the provided functions"""
        # Generate test data with 10 days of data at 10-second intervals
        self.data = generate_test_data(
            days=10,
            freq="10s",
            act_night=[0, 10],
            act_day=[10, 100],
            light_night=[0, 1],
            light_day=[500, 501],
        )

        # generate test function
        @validate_input
        def add(data):
            return data

        self.add = add

    def test_detecting_nan_values(self):
        """Test if nan data correctly raises an error"""
        # Add in 100 random NaN values to the test data
        nan_values = np.random.randint(0, len(self.data), size=100)
        data = self.data
        data.iloc[nan_values] = np.nan

        # Check decorator errors properly
        with self.assertRaises(ValueError):
            self.add(data)

    def test_all_zero_values(self):
        """Test error generated if all data is zeros"""
        data = self.data.copy()
        data.iloc[:] = 0

        with self.assertRaises(ValueError):
            self.add(data)

    def test_empty_dataframe(self):
        """Test if empty DataFrame correctly raises an error"""
        empty_data = pd.DataFrame()

        with self.assertRaises(ValueError) as context:
            self.add(empty_data)

        # Empty DataFrames are caught by the "all zeros" check
        self.assertIn("zeros", str(context.exception).lower())

    def test_non_datetime_index(self):
        """Test if DataFrame without DatetimeIndex raises an error"""
        data = self.data.copy()
        data_no_datetime = data.reset_index(drop=True)

        with self.assertRaises(TypeError) as context:
            self.add(data_no_datetime)

        self.assertIn("DatetimeIndex", str(context.exception))

    def test_missing_frequency_attribute(self):
        """Test if DataFrame without frequency attribute raises an error"""
        data = self.data.copy()
        # Create a DatetimeIndex without frequency by converting to list
        new_index = pd.to_datetime(data.index.tolist())
        data_no_freq = pd.DataFrame(data.values, index=new_index, columns=data.columns)

        with self.assertRaises(TypeError) as context:
            self.add(data_no_freq)

        self.assertIn("freq", str(context.exception).lower())

    def test_valid_dataframe_passes(self):
        """Test if valid DataFrame passes all validation checks"""
        # Should not raise any exception
        result = self.add(self.data)

        # Result should be the same as input
        pd.testing.assert_frame_equal(result, self.data)

    def test_series_with_nan(self):
        """Test if Series with NaN values raises an error"""
        series_data = pd.Series([1, 2, np.nan, 4, 5])

        @validate_input
        def process_series(data):
            return data

        with self.assertRaises(ValueError) as context:
            process_series(series_data)

        self.assertIn("NaN", str(context.exception))

    def test_series_all_zeros(self):
        """Test if Series with all zeros raises an error"""
        series_data = pd.Series([0, 0, 0, 0, 0])

        @validate_input
        def process_series(data):
            return data

        with self.assertRaises(ValueError) as context:
            process_series(series_data)

        self.assertIn("zeros", str(context.exception).lower())

    def test_series_empty(self):
        """Test if empty Series raises an error"""
        series_data = pd.Series([])

        @validate_input
        def process_series(data):
            return data

        with self.assertRaises(ValueError) as context:
            process_series(series_data)

        # Empty Series are caught by the "all zeros" check
        self.assertIn("zeros", str(context.exception).lower())

    def test_valid_series_passes(self):
        """Test if valid Series passes validation"""
        series_data = pd.Series([1, 2, 3, 4, 5])

        @validate_input
        def process_series(data):
            return data

        result = process_series(series_data)
        pd.testing.assert_series_equal(result, series_data)

    def test_kwargs_validation(self):
        """Test if decorator validates kwargs correctly"""

        @validate_input
        def process_with_kwargs(data, reference=None):
            return data, reference

        # Test with valid kwarg
        valid_series = pd.Series([1, 2, 3, 4, 5])
        result_data, result_ref = process_with_kwargs(self.data, reference=valid_series)
        pd.testing.assert_frame_equal(result_data, self.data)
        pd.testing.assert_series_equal(result_ref, valid_series)

        # Test with invalid kwarg (NaN values)
        invalid_series = pd.Series([1, np.nan, 3])
        with self.assertRaises(ValueError):
            process_with_kwargs(self.data, reference=invalid_series)

    def test_multiple_args_validation(self):
        """Test if decorator validates multiple positional arguments"""

        @validate_input
        def process_multiple(data1, data2):
            return data1, data2

        # Test with both valid
        result1, result2 = process_multiple(self.data, self.data.copy())
        pd.testing.assert_frame_equal(result1, self.data)

        # Test with second arg invalid
        data_with_nan = self.data.copy()
        data_with_nan.iloc[0] = np.nan

        with self.assertRaises(ValueError):
            process_multiple(self.data, data_with_nan)

    def test_nan_percentage_in_error_message(self):
        """Test if NaN error message includes percentage"""
        data = self.data.copy()
        # Add NaN values to first 10 rows (affects all columns)
        data.iloc[:10] = np.nan

        with self.assertRaises(ValueError) as context:
            self.add(data)

        error_msg = str(context.exception)
        # Check that percentage is mentioned
        self.assertIn("%", error_msg)
        # Check that NaN count is mentioned
        self.assertIn("NaN values", error_msg)

    def test_non_pandas_objects_ignored(self):
        """Test if decorator ignores non-pandas objects"""

        @validate_input
        def process_mixed(data, scalar_value, dict_value):
            return data, scalar_value, dict_value

        # Should not raise error for non-pandas arguments
        result_data, result_scalar, result_dict = process_mixed(
            self.data, 42, {"key": "value"}
        )

        pd.testing.assert_frame_equal(result_data, self.data)
        self.assertEqual(result_scalar, 42)
        self.assertEqual(result_dict, {"key": "value"})


if __name__ == "__main__":
    unittest.main()
