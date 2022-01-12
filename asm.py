import numpy as np
import pandas as pd


class AutoStringMapper:
    def __init__(self, from_column: any, to_column: any, ignore_case: bool = True) -> None:
        """
        Initiates an AutoStringMapper object with two string lists, series or
        np.arrays and creating a similarity matrix based on their string
        representations using the levenshtein distance. Use the get_mapping
        function of this object to then retrieve a mapping from it.

        Args:
            from_column (list, pandas.Series, np.ndarray): list of entries to
                map from
            to_column (list, pandas.Series, np.ndarray): list of entries to map
                to

        """
        from_column = self.clean_column(from_column, "from_column")
        to_column = self.clean_column(to_column, "to_column")

        unique_from_column = from_column.drop_duplicates().reset_index(drop=True)
        unique_to_column = to_column.drop_duplicates().reset_index(drop=True)

        len_from_column = unique_from_column.shape[0]
        len_to_column = unique_to_column.shape[0]

        maxlen_from_column = unique_from_column.str.len().max()
        maxlen_to_column = unique_to_column.str.len().max()

        (
            from_column_combinations,
            to_column_combinations,
        ) = self.create_combinations(unique_from_column, unique_to_column)

        if ignore_case:
            from_column_combinations = from_column_combinations.str.lower()
            to_column_combinations = to_column_combinations.str.lower()

        levenshtein_array = self.create_levenshtein_array(
            from_column_combinations,
            to_column_combinations,
            len_from_column,
            len_to_column,
            maxlen_from_column,
            maxlen_to_column,
        )

        self.distance_matrix = pd.DataFrame(
            levenshtein_array[:, maxlen_from_column - 1, maxlen_to_column - 1].reshape([len_to_column, len_from_column])
        )

        maxlen_matrix = self.create_maxlen_matrix(unique_from_column, unique_to_column)

        self.similarity_matrix = 1 - (self.distance_matrix / maxlen_matrix)

        self.similarity_matrix.index = unique_to_column.to_list()
        self.similarity_matrix.columns = unique_from_column.to_list()

    @staticmethod
    def determine_unused_row_name(index: pd.Index) -> str:
        """
        Searches and index for a suitable row name for the max column and
        creates one that is not in the index by appending.

        Args:
            index (pd.Index): index to be checked for the row name

        Returns:
            str: a row name that is not in the index

        """
        row_name = "max"
        while row_name in index:
            row_name += "max"
        return row_name

    def get_mapping(
        self,
        similarity_threshold: float = 0.0,
        relationship_type: str = "1:n",
        data_type: str = "dict",
    ) -> dict:
        """
        Function to retrieve a mapping using the similarity matrix

        Args:
            similarity_threshold (float): threshold which decides how
                similar two strings need to be in order to be included
                into the mapping and not as np.nan
            relationship_type (str): determines whether the mapping is a "1:n"
                or a "1:1" relationship


        Returns:
            dict: dictionary with the mapping from the "from" to the "to" column

        Raises:
            ValueError: if similarity_threshold is not between 0 and 1 or if
                relationship_type is not "1:1" or "1:n" or if data_type is not
                "dict", "series" or "frame"

        """
        if similarity_threshold < 0.0 or similarity_threshold > 1.0:
            raise ValueError("Parameter similarity_threshold must be between 0 and 1")

        mapping = self.similarity_matrix.idxmax(axis=0)

        if relationship_type == "1:1":

            max_row_name = determine_unused_row_name(index=self.similarity_matrix.index)

            self.similarity_matrix.loc[max_row_name] = self.similarity_matrix.max(axis=0)

            self.similarity_matrix.drop([max_row_name], inplace=True)

            self.similarity_matrix.sort_values(by=max_row_name, ascending=False, axis=1, inplace=True)

            duplication_mask = self.similarity_matrix.idxmax(axis=0).duplicated()

        elif relationship_type == "1:n":

            duplication_mask = False

        else:

            raise ValueError("Parameter relationship_type must be " "1:1" " or " "1:n" "")

        similarity_threshold_mask = self.similarity_matrix.max(axis=0) >= similarity_threshold

        net_mask = similarity_threshold_mask | duplication_mask

        mapping = mapping.where(net_mask, np.nan)

        if data_type == "dict":

            return mapping.to_dict()

        elif data_type == "series":

            return mapping

        elif data_type == "frame":

            mapping = pd.DataFrame(mapping).reset_index()
            mapping.columns = ["from", "to"]
            return mapping

        else:
            raise ValueError("Parameter data_type must be " "dict" " or " "series" " or " "frame" "")

    @staticmethod
    def clean_column(column: any, column_name: str) -> pd.Series:
        """
        Cleans either of the from / to columns to be a pandas Series of type str.

        Args:
            column (list, pandas.Series, np.ndarray): column to be cleaned
            column_name (str): specifying whether this is the from or the to column

        Returns:
            pandas.Series: converted to type str

        Raises:
            ValueError: if not of any of the expected types

        """
        if type(column) == np.ndarray:
            column = pd.Series(column)
        elif type(column) == list:
            column = pd.Series(column)
        elif type(column) == pd.Series:
            pass
        else:
            raise ValueError(f"{column_name} not of type numpy.ndarray, pandas.Series or list")
        return column.astype(str)

    @staticmethod
    def create_combinations(from_column: pd.Series, to_column: pd.Series):
        """
        Creates all combinations of strings in the from column with all strings
        in the to column returning it as two pandas.Series to be interpreted
        together.

        Args:
            from_column (pandas.Series): column that is mapped from
            to_column (pandas.Series): column that is mapped to

        Returns:
            tuple: tuple including all the combinations with the from_column strings
                as the first entry and the to_column strings as the second
        """

        len_to_column = to_column.shape[0]
        len_from_column = from_column.shape[0]

        from_column_combinations = pd.Series(from_column.tolist() * len_to_column)

        to_column_combinations = []
        for element in to_column.tolist():
            list_element = [element] * len_from_column
            to_column_combinations = to_column_combinations + list_element

        to_column_combinations = pd.Series(to_column_combinations)

        return from_column_combinations, to_column_combinations

    @staticmethod
    def create_levenshtein_array(
        from_column: pd.Series,
        to_column: pd.Series,
        len_from_column: int,
        len_to_column: int,
        maxlen_from_column: int,
        maxlen_to_column: int,
    ) -> np.ndarray:
        """
        Creates a levenshtein matrix for alle from-to-string-combinations at the
        same time in a vectorized fashion.

        Args:
            from_column (pandas.Series): combinations of the from_column (needs
                to be read together with the to_column)
            to_column (pandas.Series): combinations of the to_column (needs to
                be read together with the from_column)
            len_from_column (int): number of elements in the from_column
            len_to_column (int): number of elements in the to_column
            maxlen_from_column (int): number of characters in the longest str
                of the from_column
            maxlen_to_column (int): number of characters in the longest str
                of the to_column

        Returns:
            np.ndarray: 3-dimensional array that includes the 2-dimensionl
            levenshtein array for alle from-to-string-combinations

        """

        levenshtein_array = np.zeros(
            [len_from_column * len_to_column, maxlen_from_column, maxlen_to_column],
            "int16",
        )

        for from_column_index in range(maxlen_from_column):
            for to_column_index in range(maxlen_to_column):

                if from_column_index == 0:

                    insertion = np.array([np.iinfo("int16").max] * len_from_column * len_to_column)

                else:

                    insertion = levenshtein_array[:, from_column_index - 1, to_column_index] + (
                        ~pd.isnull(from_column.str[from_column_index])
                    ).astype("int16")

                if to_column_index == 0:

                    deletion = np.array([np.iinfo("int16").max] * len_from_column * len_to_column)

                else:

                    deletion = levenshtein_array[:, from_column_index, to_column_index - 1] + (
                        ~pd.isnull(to_column.str[to_column_index])
                    ).astype("int16")

                if from_column_index == 0 or to_column_index == 0:

                    replacement = np.array([np.iinfo("int16").max] * len_from_column * len_to_column)

                    if from_column_index == 0 and to_column_index == 0:

                        comparison = from_column.str[from_column_index] != to_column.str[to_column_index]
                        replacement = comparison.astype("int16")

                else:

                    comparison = from_column.str[from_column_index] != to_column.str[to_column_index]
                    replacement = levenshtein_array[:, from_column_index - 1, to_column_index - 1] + comparison.astype("int16")

                levenshtein_array[:, from_column_index, to_column_index] = np.array([insertion, deletion, replacement]).min(axis=0)

        return levenshtein_array

    @staticmethod
    def create_maxlen_matrix(from_column: pd.Series, to_column: pd.Series) -> pd.DataFrame:
        """
        Creates a matrix which contains the maximum of the string lengths of all
        from-to-combination pairs.

        Args:
            from_column (pandas.Series): from_column strings
            to_column (pandas.Series): to_column strings

        Returns:
            pandas.DataFrame: matrix which contains the maximum length of the
                pairs

        """

        from_column_len = from_column.shape[0]
        to_column_len = to_column.shape[0]

        divisor_frame_from = pd.concat([from_column.str.len()] * to_column_len, axis=1).T

        # get rid of row and column index
        divisor_frame_from = divisor_frame_from.T.reset_index(drop=True).T
        divisor_frame_from.reset_index(drop=True, inplace=True)

        divisor_frame_to = pd.concat([to_column.str.len()] * from_column_len, axis=1)

        # get rid of row and column index
        divisor_frame_to = divisor_frame_to.T.reset_index(drop=True).T
        divisor_frame_to.reset_index(drop=True, inplace=True)

        maxlen_matrix = pd.concat([divisor_frame_from, divisor_frame_to]).groupby(level=0).max().astype("float64")

        return maxlen_matrix
