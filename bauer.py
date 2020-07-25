""" Completion via the Bauer method, using the Schur algorithm
Copyright (C) 2020 Finn Voichick

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY 
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A 
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with 
this program.  If not, see <https://www.gnu.org/licenses/>.
"""
from math import sqrt

import numpy as np

from LPoly import LPoly, Id, LAlg


# based on doi:10.1023/A:1018915407202
def completion_from_bauer(p: LPoly) -> LAlg:
    """
    Find a Low Algebra element g such that the identity components are given by
    the input p.
    """
    n = p.degree
    poly = (Id - (p * ~p)).coefs
    xpoly = LPoly(schur(poly[n:]), -n)
    return LAlg(p, xpoly)


# based on doi:10.1016/S0024-3795(96)00517-4
def schur(v: list) -> list:
    """
    Find the final row of the Cholesky decomposition of a Toeplitz matrix.

    Parameters
    ----------
    v : list
        A list of floats such that the Toeplitz matrix formed from these floats
        is positive definite

    Returns
    -------
    list
        The bottom row of the Cholesky decomposition of the Toeplitz matrix
    """
    first = v[0]
    sqrt_first = sqrt(first)
    scaled = [x / first for x in v]
    g = np.array([scaled, scaled])
    g[1, 0] = 0
    result = []
    for _ in range(len(v)):
        rho = -g[1, 0] / g[0, 0]
        scalar = 1 / sqrt((1 - rho) * (1 + rho))
        hg = scalar * np.array([[1, rho], [rho, 1]]) @ g
        result.append(hg[0, -1] * sqrt_first)
        g = np.row_stack((hg[0, :-1], hg[1, 1:]))
    result.reverse()
    return result
