# -*- coding: utf-8 -*-

"""This script tests that coefficients of the Kolmogorov-Zurbenko filter are calculated correctly.

Coefficients of the Kolmogorov-Zurbenko filter are the coefficients of the polynomial function
:math:`(1+z+...+z^{m-1})^k`. So to check their correctness, one just has to check that equation

 .. math::

    (1 + z + \\cdots + z^{m-1})^k = \\sum_{s=-k(m-1)/2}^{k(m-1)/2} a_s^{m,k} \\cdot z^{s+k(m-1)/2}

holds for any arbitrary value of :math:`z`.


"""

from kolzur_filter import _kz_coeffs
import numpy as np
import matplotlib.pyplot as plt

__author__ = 'Mathieu Schopfer'
__version__ = '2017-03-27'

m = 51
k = 3
f = m**k

# Calculate normalised coeffients from implemented function
coeffs = _kz_coeffs(m, k)*m**k

# Defines a set of values
z = np.random.random(20)*np.random.randint(1, 2)

# Calculate polynomial function values from z
pz = np.power(np.sum(np.array([np.power(z, n) for n in range(m)]), axis=0), k)
assert pz.shape == (z.size,)

# Calculate poylnamial function values from the KZ coefficients
pc = np.sum(np.array([np.power(z, n) for n in range(k*(m-1)+1)])*coeffs[:, np.newaxis], axis=0)
assert pz.shape == (z.size,)

print(np.allclose(pz, pc))
