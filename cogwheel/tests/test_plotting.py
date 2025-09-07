"""Tests for the `plotting` module."""

from unittest import TestCase, main

import numpy as np
import pandas as pd

from cogwheel import gw_plotting


class MultiCornerPlotTestCase(TestCase):
    """Tests for making a corner plot of multiple sets of samples."""

    def test_multicornerplot(self):
        """
        Generate random weighted and unweighted samples and plot them.
        """
        rng = np.random.default_rng()
        n_samples = 1000

        weighted_samples = pd.DataFrame({
            'd_luminosity': rng.normal(1000, 100, n_samples),
            'cosiota': rng.uniform(-1, 1, n_samples),
            'weights': rng.exponential(size=n_samples)
        })
        unweighted_samples = weighted_samples.drop(columns='weights')

        mcp = gw_plotting.MultiCornerPlot(
            {'Weighted': weighted_samples, 'Unweighted': unweighted_samples},
            tail_probability=1e-3)

        mcp.plot()
        mcp.scatter_points(weighted_samples[:3])

if __name__ == '__main__':
    main()
