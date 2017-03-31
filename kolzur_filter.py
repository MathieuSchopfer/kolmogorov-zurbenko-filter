# -*- coding: utf-8 -*-

# kolzur_filter module - NumPy implementation of the Kolmogorov-Zurbenko filter
# Copyright (C) 2017  Mathieu Schopfer
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


"""Numpy implementation of the Kolmogorov-Zurbenko filter

https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Zurbenko_filter

.. todo:: Implement the KZ adaptive filter.
"""

import numpy as np

__author__ = 'Mathieu Schopfer'
__version__ = '2017-03-31'


def sliding_window(arr, window):
    """Apply a sliding window on a numpy array.

    :param numpy.ndarray arr: An array of shape `(n1, ..., nN)`
    :param int window: Window size.
    :return: A :class:`numpy.ndarray` of shape `(n1, ..., nN-window+1, window)`.

    .. seealso:: Source http://stackoverflow.com/a/6811241/3849212

    Usage (1D):

    .. doctest::

        >>> arr = np.arange(10)
        >>> arrs = sliding_window(arr, 5)
        >>> arrs.shape
        (6, 5)
        >>> print(arrs[0])
        [0 1 2 3 4]
        >>> print(arrs[1])
        [1 2 3 4 5]

    Usage (2D):

    .. doctest::

        >>> arr = np.arange(20).reshape(2, 10)
        >>> arrs = sliding_window(arr, 5)
        >>> arrs.shape
        (2, 6, 5)
        >>> print(arrs[0, 0])
        [0 1 2 3 4]
        >>> print(arrs[0, 1])
        [1 2 3 4 5]
    """

    # Advanced numpy tricks
    shape = arr.shape[:-1] + (arr.shape[-1]-window+1, window)
    strides = arr.strides + (arr.strides[-1],)
    return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)


def _kz_coeffs(m, k):
    """Calculate coefficients of the Kolmogorov–Zurbenko filter

    :return: A :class:`numpy.ndarray` of size `k*(m-1)+1`

    This functions returns the normlalised coefficients :math:`a_s^{m,k}/m^k`.

    .. rubric:: Coefficients definition

    A definition of the Kolmogorov–Zurbenko filter coefficients is provided in `this article
    <http://onlinelibrary.wiley.com/doi/10.1002/wics.71/pdf>`_. Coefficients :math:`a_s^{m,k}`  are
    the coefficients of the polynomial function:

    .. math::

        (1 + z + \\cdots + z^{m-1})^k = \\sum_{s=-k(m-1)/2}^{k(m-1)/2} a_s^{m,k} \\cdot z^{s+k(m-1)/2}

    The :math:`a_s^{m,k}` coefficients are calculated by iterating over :math:`k`.

    .. rubric:: Calculation example for m=5 and k=3

    Let us define the polynomial function

    .. math::

        P(z) = 1 + z + z^2 + z^3 + z^4.

    At :math:`k=1`, the coefficients :math:`a_s^{m,k}=a_s^{1,5}` are that of :math:`P(z)`, 

    .. math::

        \\left(\\begin{matrix}
        1 & 1 & 1 & 1 & 1
        \\end{matrix}\\right).

    At :math:`k=2`, we want to calculate the coefficients of polynomial function :math:`P(z)\\cdot P(z)`, of degree 8.
    First, we calculate the polynomial functions :math:`P(z)`, :math:`zP(z)`, :math:`z^2P(z)` and :math:`z^3P(z)` and
    then sum them.

    Let us represent the coefficients of these functions in a table, with monomial elements in columns:

    .. math::

        \\begin{array}{r|ccccccccc}
        & z^0 & z^1 & z^2 & z^3 & z^4 & z^5 & z^6 & z^7 & z^8 \\\\
        \\hline
        P(z)    & 1 & 1 & 1 & 1 & 1 & 0 & 0 & 0 & 0 \\\\
        zP(z)   & 0 & 1 & 1 & 1 & 1 & 1 & 0 & 0 & 0 \\\\
        z^2P(z) & 0 & 0 & 1 & 1 & 1 & 1 & 1 & 0 & 0 \\\\
        z^3P(z) & 0 & 0 & 1 & 1 & 1 & 1 & 1 & 1 & 0 \\\\
        z^4P(z) & 0 & 0 & 0 & 1 & 1 & 1 & 1 & 1 & 1 \\\\
        \\hline
        \\mathrm{Sum} & 1 & 2 & 3 & 4 & 5 & 4 & 3 & 2 & 1
        \\end{array}

    At :math:`k=3`, we want to calculate the coefficients of polynomial function :math:`P(z)\\cdot P(z)^2`, of degree
    12. We use the same representation:

    .. math::

        \\begin{array}{r|ccccccccccccc}
        & z^0 & z^1 & z^2 & z^3 & z^4 & z^5 & z^6 & z^7 & z^8 & z^9 & z^{10} & z^{11} & z^{12} \\\\
        \\hline
        P(z)^2    & 1 & 2 & 3 & 4 & 5 & 4 & 3 & 2 & 1 & 0 & 0 & 0 & 0 \\\\
        zP(z)^2   & 0 & 1 & 2 & 3 & 4 & 5 & 4 & 3 & 2 & 1 & 0 & 0 & 0 \\\\
        z^2P(z)^2 & 0 & 0 & 1 & 2 & 3 & 4 & 5 & 4 & 3 & 2 & 1 & 0 & 0 \\\\
        z^3P(z)^2 & 0 & 0 & 0 & 1 & 2 & 3 & 4 & 5 & 4 & 3 & 2 & 1 & 0 \\\\
        z^4P(z)^2 & 0 & 0 & 0 & 0 & 1 & 2 & 3 & 4 & 5 & 4 & 3 & 2 & 1 \\\\
        \\hline
        \\mathrm{Sum} & 1 & 3 & 6 & 10 & 15 & 18 & 19 & 18 & 15 & 10 & 6 & 3 & 1
        \\end{array}

    .. doctest::

        >>> c = _kz_coeffs(3, 1)
        >>> print(c)
        [ 0.33333333  0.33333333  0.33333333]
        >>> c = _kz_coeffs(3, 2)
        >>> print(c*3**2)
        [ 1.  2.  3.  2.  1.]
        >>> c = _kz_coeffs(5, 3)
        >>> print(c*5**3)
        [  1.   3.   6.  10.  15.  18.  19.  18.  15.  10.   6.   3.   1.]
    """

    # Coefficients at degree one
    coef = np.ones(m)

    # Iterate k-1 times over coefficients
    for i in range(1, k):

        t = np.zeros((m, m+i*(m-1)))
        for km in range(m):
            t[km, km:km+coef.size] = coef

        coef = np.sum(t, axis=0)

    assert coef.size == k*(m-1)+1

    return coef/m**k


def _kz_prod(data, coef, m, k, t=None):

    n = data.size
    data = sliding_window(data, k*(m-1)+1)
    assert data.shape == (n-k*(m-1), len(coef))

    # Restrict KZ product calculation to provided indices
    if t is not None:
        data = data[t]
        assert data.shape == (len(t), len(coef))

    return data*coef


def _kz_sum(data, coef):

    knan = np.isnan(data)

    # Handle missing values if any
    if np.any(knan):

        coef = np.ma.MaskedArray(np.broadcast_to(coef[np.newaxis, :], data.shape), mask=knan)
        coef = np.sum(coef, axis=-1)

        data = np.nansum(data, axis=-1)

        # Restore nan were data are missing
        data[coef.mask] = np.nan

        # Divide by coefficients sum, which may not be 1
        k = np.logical_not(coef.mask)
        data[k] = data[k]/coef[k]

        return data

    else:
        return np.sum(data, axis=-1)


def kz_filter(data, m, k):
    """Kolmogorov-Zurbenko fitler

    :param numpy.ndarray data: A 1-dimensional numpy array of size `N`. Any missing value should be set to ``np.nan``.
    :param int m: Filter window width.
    :param int k: Filter degree.
    :return: A :class:`numpy.ndarray` of size `N-k*(m-1)`

    Given a time series :math:`X_t, t \\in \\{0, 1, \\cdots, N-1\\}`, the Kolmogorov-Zurbenko fitler is defined for
    :math:`t \\in \\{\\frac{k(m-1)}{2}, \\cdots, N-1-\\frac{k(m-1)}{2}\\}` by

    .. math::

        KZ_{m,k}[X_t] = \\sum_{s=-k(m-1)/2}^{k(m-1)/2} \\frac{a_s^{m,k}}{m^k} \\cdot X_{t+s}

    Definition of coefficients :math:`a_s^{m,k}` is given in :func:`_kz_coeffs`.
    """

    coef = _kz_coeffs(m, k)
    data = _kz_prod(data, coef, m, k)

    return _kz_sum(data, coef)


def kzft(data, nu, m, k, t=None, dt=1.):
    """Kolmogorov-Zurbenko Fourier transform filter

    :param numpy.ndarray data: A 1-dimensional numpy array of size `N`. Any missing value should be set to ``np.nan``.
    :param list-like nu: Frequencies, length `Nnu`.
    :param int m: Filter window width.
    :param int k: Filter degree.
    :param list-like t: Calculation indices, of length `Nt`. If provided, KZFT filter will be calculated only for values
      ``data[t]``. Note that the KZFT filter can only be calculated for indices in the range [k(m-1)/2, (N-1)-k(m-1)/2].
      Trying to calculate the KZFT out of this range will raise an `IndexError`. `None`, calculation will happen over
      the whole calculable range.
    :param float dt: Time step, if not 1.
    :return: A :class:`numpy.ndarray` of shape `(Nnu, Nt)` or `(Nnu, N-k(m-1))` if `t` is `None`.
    :raise IndexError: If `t` contains one or more indices out of the calculation range. See documentation of keyword
      argument `t`.

    Given a time series :math:`X_t, t \\in \\{0, 1, \\cdots, N-1\\}`, the Kolmogorov-Zurbenko Fourier transform filter
    is defined for :math:`t \\in \\{\\frac{k(m-1)}{2}, \\cdots, N-1-\\frac{k(m-1)}{2}\\}` by
    
    .. math::

        KZFT_{m,k,\\nu}[X_t] = \\sum_{s=-k(m-1)/2}^{k(m-1)/2} \\frac{a_s^{m,k}}{m^k} \\cdot X_{t+s} \\cdot
        e^{-2\\pi i\\nu s}
    """

    if not dt == 1.:
        nu = np.asarray(nu)*dt
        m = int((m-1)/dt+1)
        if not m%2:
            m += 1

    if t is not None:
        w = int(k*(m-1)/2)
        t = np.asarray(t)-w
        if np.any(t < 0) or np.any(t > (data.size-1-2*w)):
            raise IndexError('Inpunt calculation indices are out of range. Calculation indices should be in the range '
                             '[k*(m-1)/2, (N-1)-k*(m-1)/2], hence [{}, {}] in the present case.'
                             .format(w, data.size-1-w))

    coef = _kz_coeffs(m, k)
    data = _kz_prod(data, coef, m, k, t=t)

    nu = np.asarray(nu)
    s = k*(m-1)/2
    s = np.arange(-s, s+1)
    s = np.exp(-1j*2*np.pi*nu[:, np.newaxis]*s)

    data = data[np.newaxis]*s[:, np.newaxis]

    return _kz_sum(data, coef)


def kzp(data, nu, m, k, dt=1.):
    """Kolmogorov-Zurbenko periodogram

    :param numpy.ndarray data: A 1-dimensional numpy array of size `N`. Any missing value should be set to ``np.nan``.
    :param list-like nu: Frequencies, length `Nnu`.
    :param int m: Filter window width.
    :param int k: Filter degree.
    :param float dt: Time step, if not 1.
    :return: A :class:`numpy.ndarray` os size `Nnu`.

    Given a time series :math:`X_t, t \\in \\{0, 1, \\cdots, N-1\\}`, the Kolmogorov-Zurbenko periodogram is defined by

    .. math::

        KZP_{m,k}(\\nu) = \\sqrt{\\sum_{h=0}^{T-1} \\lvert 2 \\cdot  KZFT_{m,k,\\nu}[X_{hL+k(m-1)/2}] \\rvert ^2}

    where :math:`L=(N-w)/(T-1)` is the distance between the beginnings of two successive intervals, :math:`w` being the
    calculation window width of the :func:`kzft` and :math:`T` the number of intervals. 

    The assumption was made that :math:`L \\ll w \\ll N`, implying that the intervals overlap.
    """

    if not dt == 1.:
        nu = nu*dt
        m = int((m-1)/dt+1)
        if not m%2:
            m += 1

    # w is the width of the KZFT. As m is odd, k*(m-1) is always even, so w is always odd.
    w = k*(m-1)+1

    # Distance between two successve intervals
    l = int(m/10)
    nt = int((data.size-w)/l+1)

    # Calculation indices
    l = np.arange(nt-1)*l+k*(m-1)/2
    l = np.floor(l).astype(int)

    return np.sqrt(np.nanmean(np.square(2*np.abs(kzft(data, nu, m, k, t=l))), axis=-1))
