import numpy as np

def check_ndigits(x, ndigits=0):
    rounded_x = round_vec(x, ndigits)
    if np.any(rounded_x == 0):
        raise Warning("Is argument `ndigits` too small?")
    pass

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


def calc_signif_digits(x):
    n_signif_digits = get_n_signif_digits(x)
    rounded_x = round_vec(x, n_signif_digits)
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
    # 0.198 -> 0.2, (NOT 0.20)
    if isinstance(n_signif_digits, int) or isinstance(n_signif_digits, float):
        n_signif_digits = np.repeat(n_signif_digits, repeats=len(x))
    out = [
        f"%.{ndigits}f" % round(val, ndigits)
        for val, ndigits in zip(x, n_signif_digits)
    ]
    return out

def round_vec(x, digits=0):
    n = len(x)
    if isinstance(digits, int) or isinstance(digits, float):
        digits = np.repeat(digits, repeats=len(x))
    # if len(digits) == 1:
    #     digits = np.repeat(digits, repeats=n)
    if len(digits) > 1 and len(digits) != n:
        raise Exception("len(x) not equal to len(digits)")
    
    rounded_x = np.array(
        [np.round(val, ndigits) for val, ndigits in zip(x, digits)]
    )
    return rounded_x

def latex_range(best, min, max, ndigits = 'auto'):
    """Make latex notation for best value and confidence boundaries.

    Parameters
    ----------
    best : np.ndarray
        best value
    min : np.ndarray
        lower boundary value
    max : np.ndarray
        upper boundary value
    ndigits : str,int,np.ndarray
        number of digits after decimal point, by default 'auto'

    Returns
    -------
    np.ndarray
        string vector
    
    Examples
    -------
    >>> x = np.random.uniform(1, 10, size=3)
    >>> err_lo = np.random.uniform(0, 1, size=3)
    >>> err_up = np.random.uniform(0, 1, size=3)
    >>> xmin = x - err_lo
    >>> xmax = x + err_up
    >>> latex_range(x, xmin, xmax)
    array(['$6.49_{-0.75}^{+0.06}$', '$5.966_{-0.717}^{+0.008}$',
       '$2.3_{-0.8}^{+0.7}$'], dtype='<U25')
    >>> latex_range(x, xmin, x + err_lo)
    array(['$6.5\\pm0.7$', '$6.0\\pm0.7$', '$2.3\\pm0.8$'], dtype='<U11')
    """    
    n = len(best)
    x = best
    xmin = min
    xmax = max
    if ndigits == 'auto':
        err_min = np.min([xmax-x, x - xmin], axis=-0)
        ndigits = calc_signif_digits(err_min)
    else:
        # ndigits = np.repeat(1, repeats=n)
        ndigits = 1

    lo = round_vec(x-xmin, ndigits)
    up = round_vec(xmax-x, ndigits)

    if len(ndigits) == 1:
        check_ndigits(np.concatenate([lo,up]), ndigits)
    
    if np.all(lo == up):
        out_x = format_number(x, ndigits)
        out_pm = format_number(lo, ndigits)
        out = ["$" + x_i + "\\pm" +  x_i_err + "$" for x_i, x_i_err in zip(out_x, out_pm)]
    else:
        out_x = format_number(x, ndigits)
        out_lo = format_number(lo, ndigits)
        out_up = format_number(up, ndigits)
        out = ["$" + x_i + "_{-" +  x_i_lo + "}^{+" + x_i_up + "}$" for x_i, x_i_lo, x_i_up in zip(out_x, out_lo, out_up)]
    
    return np.array(out)
