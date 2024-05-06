import numpy as np


def get_first_digit(x):
    tmp_rounded = np.round(np.abs(np.log10(np.abs(x))), 0)
    first_digit = np.abs(np.power(10, tmp_rounded) * x)
    first_digit = np.floor_divide(first_digit, 1)
    return first_digit


def get_n_signif_digits(x):
    first_digits = get_first_digit(x)
    n_signif_digits = np.abs(np.ceil(-np.log10(np.abs(x))))
    n_signif_digits = np.where(first_digits == 1, n_signif_digits + 1, n_signif_digits)
    n_signif_digits[np.abs(x) > 2] = 0
    return n_signif_digits.astype(int)


def calc_n_digits(x):
    n_signif_digits = get_n_signif_digits(x)
    rounded_x = np.array(
        [np.round(val, ndigits) for val, ndigits in zip(x, n_signif_digits)]
    )
    n_signif_digits = np.where(
        get_n_signif_digits(rounded_x) < n_signif_digits,
        n_signif_digits - 1,
        n_signif_digits,
    )
    return n_signif_digits


def format_number(x: np.array, n_signif_digits=0):
    """Format number

    Parameters
    ----------
    x : np.array
        float/double number array
    n_signif_digits : int, optional
        number of the significant digits, by default 0

    Returns
    -------
    np.array
        formatted array
    """    
    out = [
        f"%.{ndigits}f" % round(val, ndigits)
        for val, ndigits in zip(x, n_signif_digits)
    ]
    return out
