"""
Define class ``TwoSquircularMapping`` to map a disk to a square
without resorting to polar coordinates.
Useful for samplers that do not support periodic parameters.
"""
import numpy as np


class TwoSquircularMapping:
    """
    2-squircular mapping between a disk (u, v): u^2 + v^2 <= 1
    and a square (x, y): -1 <= x,y <= 1.
    Reference: https://arxiv.org/pdf/1709.07875.pdf
    """
    @staticmethod
    def square_to_disk(x, y):
        """Return (u, v) on the disk."""
        factor = (1 + (x*y)**2) ** -0.5
        return factor * x, factor * y

    @staticmethod
    def disk_to_square(u, v):
        """Return (x, y) on the square."""
        if u == 0 or v == 0:
            return u, v

        inverse_factor = np.sqrt((0.5 - np.sqrt(0.25-(u*v)**2))) / np.abs(u*v)
        return inverse_factor * u, inverse_factor * v

    @staticmethod
    def jacobian_determinant(x, y):
        """
        Return |∂(u,v) / ∂(x,y)|.
        Prior on the square (x, y) that yields uniform on the disk (u, v).
        """
        x2y2 = (x*y) ** 2
        return (1 - x2y2) / (1 + x2y2)**2
