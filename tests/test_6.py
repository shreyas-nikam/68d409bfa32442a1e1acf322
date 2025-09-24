import pytest
from definition_95da65620b844ac18f803abb512dea59 import plot_loss_curves
import matplotlib.pyplot as plt
import numpy as np

def generate_dummy_history(loss, val_loss):
    return {'loss': loss, 'val_loss': val_loss}

def is_plot_visible():
    return plt.fignum_exists(1)

def clear_plot():
    plt.close()

@pytest.fixture(autouse=True)
def cleanup():
    clear_plot()

def test_plot_loss_curves_valid_history():
    history = generate_dummy_history([1, 0.5, 0.25], [1.1, 0.6, 0.3])
    plot_loss_curves(history)
    assert is_plot_visible()

def test_plot_loss_curves_empty_history():
    history = generate_dummy_history([], [])
    plot_loss_curves(history)
    assert is_plot_visible()

def test_plot_loss_curves_unequal_loss_val_loss():
    history = generate_dummy_history([1, 0.5], [1.1, 0.6, 0.3])
    with pytest.raises(ValueError):
        plot_loss_curves(history)

def test_plot_loss_curves_non_dict_input():
    with pytest.raises(TypeError):
        plot_loss_curves([1, 2, 3])

def test_plot_loss_curves_missing_keys():
    history = {'loss': [1, 0.5, 0.25]}
    with pytest.raises(KeyError):
        plot_loss_curves(history)
