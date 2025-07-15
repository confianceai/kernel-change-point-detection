import matplotlib.pyplot as plt


def fig_ax(figsize=(15, 5), dpi=150):
    """Generate a (matplotlib) figure and ax objects with given size."""
    return plt.subplots(figsize=figsize, dpi=dpi)


def get_sum_of_cost(algo, n_bkps) -> float:
    """Calculate the sum of costs for the change points `bkps`.

    Utility function used to do penalized variable selection and obtain
    a final list of anomaly time points.
    """
    bkps = algo.predict(n_bkps=n_bkps)
    return algo.cost.sum_of_costs(bkps)
