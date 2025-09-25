import pytest
import pandas as pd
from unittest.mock import MagicMock # MagicMock is part of unittest.mock, but pytest-mock provides 'mocker' fixture which is convenient
import os

# Keep a placeholder definition_56e0d0980e2a4644a1858df9a28e485d for the import of the module.
# Keep the `your_module` block as it is. DO NOT REPLACE or REMOVE the block.
from definition_56e0d0980e2a4644a1858df9a28e485d import plot_features_pairplot

@pytest.fixture
def sample_dataframe():
    """Provides a sample pandas DataFrame for testing."""
    return pd.DataFrame({
        'numeric_col1': [1, 2, 3, 4, 5],
        'numeric_col2': [10, 20, 15, 25, 30],
        'categorical_col': ['A', 'B', 'A', 'C', 'B'],
        'other_col': [100, 200, 300, 400, 500]
    })

@pytest.fixture(autouse=True)
def mock_plot_functions(mocker):
    """Mocks seaborn and matplotlib functions to prevent actual plotting and file saving."""
    # Mock the return value of pairplot to have a .fig attribute which is also a mock
    mock_fig = mocker.MagicMock()
    mocker.patch('seaborn.pairplot', return_value=mocker.MagicMock(fig=mock_fig))
    mocker.patch('matplotlib.pyplot.savefig')
    mocker.patch('matplotlib.pyplot.close')
    mocker.patch('matplotlib.pyplot.show') # In case plt.show() is called
    mocker.patch.object(mock_fig, 'suptitle') # Mock suptitle method on the mock figure
    mocker.patch('os.makedirs') # Mock os.makedirs to avoid creating directories

def test_plot_features_pairplot_standard_case(sample_dataframe, mocker):
    """
    Tests the function with valid inputs, including hue_column and a custom title.
    Ensures that seaborn.pairplot, fig.suptitle, plt.savefig, and plt.close are called
    with the expected arguments.
    """
    df = sample_dataframe
    features_list = ['numeric_col1', 'numeric_col2']
    hue_column = 'categorical_col'
    title = 'Test Pair Plot Title'
    
    plot_features_pairplot(df, features_list, hue_column, title)

    # Check if sns.pairplot was called with the correct data, features, and hue
    mocker.patch('seaborn.pairplot').assert_called_once()
    call_args, call_kwargs = mocker.patch('seaborn.pairplot').call_args
    assert call_kwargs['data'] is df
    assert call_kwargs['vars'] == features_list
    assert call_kwargs['hue'] == hue_column
    assert 'diag_kind' in call_kwargs and call_kwargs['diag_kind'] == 'hist' # Assuming 'hist' is a common default or explicitly set

    # Check if plt.suptitle was called on the figure object
    mocker.patch('seaborn.pairplot').return_value.fig.suptitle.assert_called_once_with(title, y=1.02)
    # Check if plt.savefig was called
    mocker.patch('matplotlib.pyplot.savefig').assert_called_once_with(os.path.join("plots", "pair_plot.png"), bbox_inches='tight')
    # Check if plt.close was called to free memory
    mocker.patch('matplotlib.pyplot.close').assert_called_once()
    mocker.patch('os.makedirs').assert_called_once_with("plots", exist_ok=True)


def test_plot_features_pairplot_no_hue_no_title(sample_dataframe, mocker):
    """
    Tests the function with valid inputs, where hue_column and title are None (default values).
    Ensures that seaborn.pairplot, plt.savefig, and plt.close are called, but suptitle is not.
    """
    df = sample_dataframe
    features_list = ['numeric_col1', 'numeric_col2']
    hue_column = None
    title = None
    
    plot_features_pairplot(df, features_list, hue_column, title)

    # Check if sns.pairplot was called without the 'hue' argument
    mocker.patch('seaborn.pairplot').assert_called_once()
    call_args, call_kwargs = mocker.patch('seaborn.pairplot').call_args
    assert call_kwargs['data'] is df
    assert call_kwargs['vars'] == features_list
    assert 'hue' not in call_kwargs or call_kwargs['hue'] is None

    # Check that plt.suptitle was NOT called
    mocker.patch('seaborn.pairplot').return_value.fig.suptitle.assert_not_called()
    mocker.patch('matplotlib.pyplot.savefig').assert_called_once()
    mocker.patch('matplotlib.pyplot.close').assert_called_once()
    mocker.patch('os.makedirs').assert_called_once_with("plots", exist_ok=True)


def test_plot_features_pairplot_single_feature_list(sample_dataframe, mocker):
    """
    Tests the function with a single feature in the features_list, a valid edge case.
    Ensures plotting functions are called correctly.
    """
    df = sample_dataframe
    features_list = ['numeric_col1']
    hue_column = 'categorical_col'
    title = 'Single Feature Pair Plot'
    
    plot_features_pairplot(df, features_list, hue_column, title)

    # Check if sns.pairplot was called with a single feature
    mocker.patch('seaborn.pairplot').assert_called_once()
    call_args, call_kwargs = mocker.patch('seaborn.pairplot').call_args
    assert call_kwargs['data'] is df
    assert call_kwargs['vars'] == features_list
    assert call_kwargs['hue'] == hue_column

    mocker.patch('seaborn.pairplot').return_value.fig.suptitle.assert_called_once_with(title, y=1.02)
    mocker.patch('matplotlib.pyplot.savefig').assert_called_once()
    mocker.patch('matplotlib.pyplot.close').assert_called_once()
    mocker.patch('os.makedirs').assert_called_once_with("plots", exist_ok=True)


def test_plot_features_pairplot_empty_dataframe_with_features(mocker):
    """
    Tests the function with an empty DataFrame but requested features.
    It's expected to raise a ValueError as there's no data to plot.
    """
    df = pd.DataFrame(columns=['numeric_col1', 'numeric_col2', 'categorical_col'])
    features_list = ['numeric_col1', 'numeric_col2']
    hue_column = 'categorical_col'
    title = 'Empty DF Plot'

    with pytest.raises(ValueError, match="Cannot plot an empty DataFrame with specified features."):
         plot_features_pairplot(df, features_list, hue_column, title)

    # Assert that no plotting functions were called because of the early exit due to ValueError
    mocker.patch('seaborn.pairplot').assert_not_called()
    mocker.patch('seaborn.pairplot').return_value.fig.suptitle.assert_not_called()
    mocker.patch('matplotlib.pyplot.savefig').assert_not_called()
    mocker.patch('matplotlib.pyplot.close').assert_not_called()
    mocker.patch('os.makedirs').assert_not_called()


@pytest.mark.parametrize("df_input, features, hue, title_input, expected_error, error_message_part", [
    # Invalid df type
    ([1, 2, 3], ['numeric_col1'], None, "Title", TypeError, "df must be a pandas DataFrame."),
    ("not a df", ['numeric_col1'], None, "Title", TypeError, "df must be a pandas DataFrame."),
    # features_list not a list
    (pd.DataFrame({'a': [1]}), 'a', None, "Title", TypeError, "features_list must be a list of strings."),
    # features_list contains non-string
    (pd.DataFrame({'a': [1]}), ['a', 1], None, "Title", TypeError, "features_list must be a list of strings."),
    # Non-existent feature column
    (pd.DataFrame({'a': [1]}), ['b'], None, "Title", KeyError, "Features not found in DataFrame: ['b']"),
    # Non-existent hue column
    (pd.DataFrame({'a': [1], 'c': ['X']}), ['a'], 'd', "Title", KeyError, "Hue column 'd' not found in DataFrame."),
    # hue_column not a string or None
    (pd.DataFrame({'a': [1]}), ['a'], 123, "Title", TypeError, "hue_column must be a string or None."),
    # title not a string or None
    (pd.DataFrame({'a': [1]}), ['a'], None, 123, TypeError, "title must be a string or None."),
])
def test_plot_features_pairplot_invalid_inputs_and_types(
    df_input, features, hue, title_input, expected_error, error_message_part, mocker
):
    """
    Tests various invalid input scenarios for DataFrame, features_list, hue_column, and title.
    Ensures appropriate errors are raised and no plotting functions are called.
    """
    with pytest.raises(expected_error) as excinfo:
        plot_features_pairplot(df_input, features, hue, title_input)
    
    assert error_message_part in str(excinfo.value)
    
    # Assert that no plotting functions were called due to the error
    mocker.patch('seaborn.pairplot').assert_not_called()
    mocker.patch('seaborn.pairplot').return_value.fig.suptitle.assert_not_called()
    mocker.patch('matplotlib.pyplot.savefig').assert_not_called()
    mocker.patch('matplotlib.pyplot.close').assert_not_called()
    mocker.patch('os.makedirs').assert_not_called()