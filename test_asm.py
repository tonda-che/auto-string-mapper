from asm import AutoStringMapper
from pandas.testing import assert_frame_equal
from pandas.testing import assert_series_equal
import pandas as pd
import numpy as np
import random
import string


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
    supposed_result = np.array(
        [
            [[0, 1, 2, 3], [1, 1, 2, 3], [2, 2, 1, 2]],
            [[1, 2, 3, 4], [2, 2, 3, 4], [3, 3, 2, 3]],
            [[1, 2, 3, 3], [2, 1, 2, 2], [3, 2, 1, 1]],
            [[1, 2, 3, 3], [2, 1, 2, 2], [3, 2, 1, 1]],
        ]
    )
    assert (actual_result == supposed_result).all()


def test_mapping():
    from_column = pd.Series(["The Beauty and the Beast", "Aladdin", "Mulan", "The Lion King"])
    to_column = pd.Series(["Aladin (1992)", "Lion King (1994)", "The Beauty and the Beast (1991)", "Mulan (1998)"])
    actual_result = AutoStringMapper(from_column, to_column).get_mapping()
    supposed_result = {
        "The Beauty and the Beast": "The Beauty and the Beast (1991)",
        "Aladdin": "Aladin (1992)",
        "Mulan": "Mulan (1998)",
        "The Lion King": "Lion King (1994)",
    }
    assert actual_result == supposed_result


def test_mapping_for_frame():
    from_column = pd.Series(["The Beauty and the Beast", "Aladdin", "Mulan", "The Lion King"])
    to_column = pd.Series(["Aladin (1992)", "Lion King (1994)", "The Beauty and the Beast (1991)", "Mulan (1998)"])
    actual_result = AutoStringMapper(from_column, to_column).get_mapping(data_type="frame")
    supposed_result = pd.DataFrame(
        {
            "from": ["The Beauty and the Beast", "Aladdin", "Mulan", "The Lion King"],
            "to": ["The Beauty and the Beast (1991)", "Aladin (1992)", "Mulan (1998)", "Lion King (1994)"],
        }
    )
    assert_frame_equal(actual_result.sort_index(axis=1), supposed_result.sort_index(axis=1))


def test_mapping_for_series():
    from_column = pd.Series(["The Beauty and the Beast", "Aladdin", "Mulan", "The Lion King"])
    to_column = pd.Series(["Aladin (1992)", "Lion King (1994)", "The Beauty and the Beast (1991)", "Mulan (1998)"])
    actual_result = AutoStringMapper(from_column, to_column).get_mapping(data_type="series")
    supposed_result = pd.Series(
        {
            "The Beauty and the Beast": "The Beauty and the Beast (1991)",
            "Aladdin": "Aladin (1992)",
            "Mulan": "Mulan (1998)",
            "The Lion King": "Lion King (1994)",
        }
    )
    assert_series_equal(actual_result.sort_index(), supposed_result.sort_index())


def test_relationship_type():
    from_column = pd.Series(["The Beauty and the Beast", "Aladdin", "Mulan", "Aladin"])
    to_column = pd.Series(["Aladin (1992)", "Lion King (1994)", "The Beauty and the Beast (1991)", "Mulan (1998)"])
    actual_result = AutoStringMapper(from_column, to_column).get_mapping(relationship_type="1:1")
    supposed_result = {
        "The Beauty and the Beast": "The Beauty and the Beast (1991)",
        "Aladdin": "Lion King (1994)",
        "Mulan": "Mulan (1998)",
        "Aladin": "Aladin (1992)",
    }
    for key in supposed_result.keys():
        assert actual_result[key] == supposed_result[key] or (pd.isnull(actual_result[key]) and pd.isnull(supposed_result[key]))


def test_similarity_threshold():
    from_column = pd.Series(["The Beauty and the Beast", "Aladdin", "Mulan", "The Lion King"])
    to_column = pd.Series(["Aladin (1992)", "Lion King (1994)", "The Beauty and the Beast (1991)", "Mulan (1998)"])
    actual_result = AutoStringMapper(from_column, to_column).get_mapping(similarity_threshold=0.4)
    supposed_result = {
        "The Beauty and the Beast": "The Beauty and the Beast (1991)",
        "Aladdin": np.nan,
        "Mulan": "Mulan (1998)",
        "The Lion King": np.nan,
    }
    for key in supposed_result.keys():
        assert actual_result[key] == supposed_result[key] or (pd.isnull(actual_result[key]) and pd.isnull(supposed_result[key]))


def test_relationship_type_and_threshold():
    from_column = pd.Series(["The Beauty and the Beast", "Aladdin", "Mulan", "Aladin"])
    to_column = pd.Series(["Aladin (1992)", "Lion King (1994)", "The Beauty and the Beast (1991)", "Mulan (1998)"])
    actual_result = AutoStringMapper(from_column, to_column).get_mapping(relationship_type="1:1", similarity_threshold=0.4)
    supposed_result = {
        "The Beauty and the Beast": "The Beauty and the Beast (1991)",
        "Aladdin": np.nan,
        "Mulan": "Mulan (1998)",
        "Aladin": "Aladin (1992)",
    }
    print(actual_result)
    for key in supposed_result.keys():
        assert actual_result[key] == supposed_result[key] or (pd.isnull(actual_result[key]) and pd.isnull(supposed_result[key]))


def test_ignore_case_deactivated():
    from_column = pd.Series(["The Beauty and the Beast", "Aladdin", "Mulan", "The Lion King", "Matrix (1999)"])
    to_column = pd.Series(["Aladin (1992)", "Lion King (1994)", "The Beauty and the Beast (1991)", "Mulan (1998)", "MATRIX 1999"])
    actual_result = AutoStringMapper(from_column, to_column, ignore_case=False).get_mapping()
    supposed_result = {
        "The Beauty and the Beast": "The Beauty and the Beast (1991)",
        "Aladdin": "Aladin (1992)",
        "Mulan": "Mulan (1998)",
        "The Lion King": "Lion King (1994)",
        "Matrix (1999)": "Aladin (1992)",
    }
    assert actual_result == supposed_result


def test_ignore_case_activated():
    from_column = pd.Series(["The Beauty and the Beast", "Aladdin", "Mulan", "The Lion King", "Matrix (1999)"])
    to_column = pd.Series(["Aladin (1992)", "Lion King (1994)", "The Beauty and the Beast (1991)", "Mulan (1998)", "MATRIX 1999"])
    actual_result = AutoStringMapper(from_column, to_column, ignore_case=True).get_mapping()
    supposed_result = {
        "The Beauty and the Beast": "The Beauty and the Beast (1991)",
        "Aladdin": "Aladin (1992)",
        "Mulan": "Mulan (1998)",
        "The Lion King": "Lion King (1994)",
        "Matrix (1999)": "MATRIX 1999",
    }
    assert actual_result == supposed_result


def test_1_to_1_with_threshold_overlay_case():
    from_column = pd.Series(["Matrix (1999)", "MATRIX 1999"])
    to_column = pd.Series(["MATRIX 1999 XYZZZZ", "MTRX"])
    actual_result = AutoStringMapper(from_column, to_column, ignore_case=True).get_mapping(
        relationship_type="1:1", similarity_threshold=0.3
    )
    supposed_result = {"Matrix (1999)": "MTRX", "MATRIX 1999": "MATRIX 1999 XYZZZZ"}
    assert actual_result == supposed_result


def get_random_string(number_of_characters=20):
    random_string = "".join(random.choices(string.ascii_uppercase + string.digits, k=number_of_characters))
    return random_string


def get_random_string_array(length=100, number_of_characters=20):
    arr = []
    for index in range(length):
        arr.append(get_random_string(number_of_characters))
    return arr


def test_performance_200_to_100():
    AutoStringMapper(from_column=get_random_string_array(200), to_column=get_random_string_array(100), ignore_case=True).get_mapping()
    assert True


def test_performance_100_to_200():
    AutoStringMapper(from_column=get_random_string_array(100), to_column=get_random_string_array(200), ignore_case=True).get_mapping()
    assert True


def test_performance_200_to_200():
    AutoStringMapper(from_column=get_random_string_array(200), to_column=get_random_string_array(200), ignore_case=True).get_mapping()
    assert True
