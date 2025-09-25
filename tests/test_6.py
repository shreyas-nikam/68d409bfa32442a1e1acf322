import pytest
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans # This is included to correctly catch the ValueError from KMeans directly

from definition_b67d7fca3ab7477e8dbd4f173b1b65a8 import perform_kmeans_clustering

@pytest.mark.parametrize("data, n_clusters, random_state, expected_output_or_exception", [
    # Test Case 1: Standard functionality with numpy.ndarray (two clear clusters)
    (np.array([[1, 1], [1.5, 1.5], [5, 5], [5.5, 5.5]]), 2, 42,
     lambda result: (len(result) == 4 and all(0 <= l < 2 for l in result) and
                     (np.array_equal(np.sort(result[:2]), [0, 0]) and np.array_equal(np.sort(result[2:]), [1, 1]) or
                      np.array_equal(np.sort(result[:2]), [1, 1]) and np.array_equal(np.sort(result[2:]), [0, 0])))),

    # Test Case 2: Standard functionality with pandas.DataFrame (two clear clusters)
    (pd.DataFrame({'x': [1, 1.5, 5, 5.5], 'y': [1, 1.5, 5, 5.5]}), 2, 42,
     lambda result: (len(result) == 4 and all(0 <= l < 2 for l in result) and
                     (np.array_equal(np.sort(result[:2]), [0, 0]) and np.array_equal(np.sort(result[2:]), [1, 1]) or
                      np.array_equal(np.sort(result[:2]), [1, 1]) and np.array_equal(np.sort(result[2:]), [0, 0])))),

    # Test Case 3: Edge case - n_clusters = 1 (all points in one cluster)
    (np.array([[1, 1], [2, 2], [3, 3]]), 1, 42, np.array([0, 0, 0])),

    # Test Case 4: Edge case - n_clusters > number of samples
    # KMeans (sklearn) raises ValueError if n_clusters > n_samples.
    (np.array([[1, 1], [2, 2]]), 3, 42, ValueError), 

    # Test Case 5: Invalid input type for n_clusters
    (np.array([[1, 1], [2, 2]]), "invalid", 42, TypeError),
])
def test_perform_kmeans_clustering(data, n_clusters, random_state, expected_output_or_exception):
    if isinstance(expected_output_or_exception, type) and issubclass(expected_output_or_exception, Exception):
        with pytest.raises(expected_output_or_exception):
            perform_kmeans_clustering(data, n_clusters, random_state)
    else:
        result = perform_kmeans_clustering(data, n_clusters, random_state)
        # Check if the result is a numpy array
        assert isinstance(result, np.ndarray)
        if callable(expected_output_or_exception):
            # For cases where cluster labels might swap (0 vs 1), use a custom check
            assert expected_output_or_exception(result)
        else:
            # For cases with fixed expected output (e.g., all 0s for n_clusters=1)
            assert np.array_equal(result, expected_output_or_exception)