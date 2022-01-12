from asm import AutoStringMapper
from pandas.testing import assert_frame_equal
from pandas.testing import assert_series_equal
import pandas as pd
import numpy as np


def test_create_maxlen_matrix():
    from_column = pd.Series(["c", "dddd"])
    to_column = pd.Series(["aa", "bbb"])
    actual_result = AutoStringMapper.create_maxlen_matrix(from_column=from_column, to_column=to_column)
    supposed_result = pd.DataFrame([[2.0, 4.0], [3.0, 4.0]])
    assert_frame_equal(actual_result, supposed_result)


def test_create_combinations():
    from_column = pd.Series(["c", "dddd"])
    to_column = pd.Series(["aa", "bbb"])
    actual_result_from, actual_result_to = AutoStringMapper.create_combinations(from_column=from_column, to_column=to_column)
    supposed_result_from = pd.Series(["c", "dddd", "c", "dddd"])
    supposed_result_to = pd.Series(["aa", "aa", "bbb", "bbb"])
    assert_series_equal(supposed_result_from, actual_result_from)
    assert_series_equal(supposed_result_to, actual_result_to)


def test_determine_unused_row_name_default():
    index = pd.Index(["a", "bb", "ccc"])
    actual_result = AutoStringMapper.determine_unused_row_name(index)
    supposed_result = "max"
    assert actual_result == supposed_result


def test_determine_unused_row_name_1():
    index = pd.Index(["a", "max", "ccc"])
    actual_result = AutoStringMapper.determine_unused_row_name(index)
    supposed_result = "maxmax"
    assert actual_result == supposed_result


def test_determine_unused_row_name_2():
    index = pd.Index(["a", "maxmax", "max"])
    actual_result = AutoStringMapper.determine_unused_row_name(index)
    supposed_result = "maxmaxmax"
    assert actual_result == supposed_result


def test_clean_column_series():
    series = pd.Series(["a", "bb", 3.0])
    actual_result = AutoStringMapper.clean_column(column=series, column_name="test")
    supposed_result = pd.Series(["a", "bb", "3.0"])
    assert_series_equal(actual_result, supposed_result)


def test_clean_column_numpy():
    numpy_array = np.array(["a", "bb", 3.0])
    actual_result = AutoStringMapper.clean_column(column=numpy_array, column_name="test")
    supposed_result = pd.Series(["a", "bb", "3.0"])
    assert_series_equal(actual_result, supposed_result)


def test_clean_column_list():
    python_list = ["a", "bb", 3.0]
    actual_result = AutoStringMapper.clean_column(column=python_list, column_name="test")
    supposed_result = pd.Series(["a", "bb", "3.0"])
    assert_series_equal(actual_result, supposed_result)


def test_create_levenshtein_array():
    from_column = pd.Series(["pun", "bun", "pun", "bun"])
    to_column = pd.Series(["pant", "pant", "sun", "sun"])
    actual_result = AutoStringMapper.create_levenshtein_array(from_column, to_column, 2, 2, 3, 4)
    print(actual_result)
    supposed_result = np.array(
        [
            [[0, 1, 2, 3], [1, 1, 2, 3], [2, 2, 1, 2]],
            [[1, 2, 3, 4], [2, 2, 3, 4], [3, 3, 2, 3]],
            [[1, 2, 3, 3], [2, 1, 2, 2], [3, 2, 1, 1]],
            [[1, 2, 3, 3], [2, 1, 2, 2], [3, 2, 1, 1]],
        ]
    )
    assert (actual_result == supposed_result).all()
