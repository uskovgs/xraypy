import numpy as np
import warnings

def check_ndigits(x, ndigits=0):
    """Checking whether the number of significant digits is enough

    Parameters
    ----------
    x : array_like
        array of numbers
    ndigits : int,array_like, optional
        number of significant digits, by default 0

    Raises
    ------
    Warning
        Is the argument `ndigits` too small?
    """    
    rounded_x = round_vec(x, ndigits)
    if np.any(rounded_x == 0):
        warnings.warn("Is the argument `ndigits` too small?")
    pass

def get_first_digit(x):
    if not isinstance(x, np.ndarray):
        x = np.array([x])
    tmp_rounded = np.round(np.abs(np.log10(np.abs(x))), 0)
    first_digit = np.abs(np.power(10, tmp_rounded) * x)
    first_digit = np.floor_divide(first_digit, 1)
    return first_digit


def get_n_signif_digits(x):
    """Get the optimal number of significant digits

    Parameters
    ----------
    x : array_like
            Input data.

    Returns
    -------
    ndarray
        array of int type significant digits
    """    
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


def format_number(x, n_signif_digits=0):
    """Format number

    Parameters
    ----------
    x : array_like
        Input data
    n_signif_digits : int, optional
        number of the significant digits, by default 0

    Returns
    -------
    np.array
        formatted array
    
    Examples
    -------
    >>> format_number(0.198, calc_signif_digits(0.198))
    ['0.2']
    """    
    # 0.198 -> 0.2, (NOT 0.20)
    if not isinstance(x, np.ndarray):
        x = np.array([x])
    
    if isinstance(n_signif_digits, int) or isinstance(n_signif_digits, float):
        n_signif_digits = np.repeat(n_signif_digits, repeats=len(x))
    out = [
        f"%.{ndigits}f" % round(val, ndigits)
        for val, ndigits in zip(x, n_signif_digits)
    ]
    return out

def round_vec(x, digits=0):
    if not isinstance(x, np.ndarray):
        x = np.array([x])
    
    if isinstance(digits, int) or isinstance(digits, float):
        digits = np.repeat(digits, repeats=len(x))
    # if len(digits) == 1:
    #     digits = np.repeat(digits, repeats=n)
    if len(digits) > 1 and len(digits) != len(x):
        raise Exception("len(x) not equal to len(digits)")
    
    rounded_x = np.array(
        [np.round(val, ndigits) for val, ndigits in zip(x, digits)]
    )
    return rounded_x

def latex_range(best, min, max, ndigits = 'auto'):
    """Make latex notation for best value and confidence boundaries.

    Parameters
    ----------
    best : array_like
        best value
    min : array_like
        lower boundary value
    max : array_like
        upper boundary value
    ndigits : str,int,array_like
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
    array(['$4.326_{-0.745}^{+0.019}$', '$7.96_{-0.84}^{+0.13}$',
       '$8.56_{-0.79}^{+0.15}$'], dtype='<U25')
    
    >>> latex_range(x, xmin, x + err_lo, ndigits=2)
    array(['$5.92\\pm0.94$', '$7.85\\pm0.33$', '$6.48\\pm0.00$'], dtype='<U13')

    >>> latex_range(x, xmin, x + err_lo, ndigits='auto')
    array(['$5.9\\pm0.9$', '$7.9\\pm0.3$', '$6.4807\\pm0.0008$'], dtype='<U17')
    """    
    n = len(best)
    x = best
    xmin = min
    xmax = max

    if ndigits == 'auto':
        err_min = np.min([xmax-x, x - xmin], axis=0)
        ndigits = calc_signif_digits(err_min)

    lo = round_vec(x-xmin, ndigits)
    up = round_vec(xmax-x, ndigits)

    if isinstance(ndigits, int) or isinstance(ndigits, float):
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

def latex_range_sci(
    best, min, max, ndigits="auto", symmetry=False, base_pow=None
):
    """Make latex notation for best value and confidence boundaries

    Parameters
    ----------
    best : array_like
        best value
    min : array_like
        lower boundary value
    max : array_like
        upper boundary value
    ndigits : str,int,array_like optional
        number of digits after decimal point, by default "auto"
    symmetry : bool, optional
        if errors are symmetric show as value+/-err, by default False
    base_pow : int, optional
        set to numeric if you want to make column with the same power., by default None

    Returns
    -------
    array_like
        string vector
    
    Examples
    -------
    
    Set initial vector

    >>> import numpy as np
    >>> x = np.array([1e21, 3e22, 1e23])
    >>> err = np.array([0.1e21, 0.5e21, 0.1e23])

    >>> latex_range_sci(best = x, min = x - err, max = x + err, ndigits = 2)
    ['$1.00_{-0.10}^{+0.10}~\\times10^{21}$',
    '$0.30_{-0.01}^{+0.00}~\\times10^{23}$',
    '$1.00_{-0.10}^{+0.10}~\\times10^{23}$']

    >>> latex_range_sci(x, x - err, x + err, ndigits = 3, symmetry = True)
    ['$(1.000\\pm0.100)~\\times10^{21}$',
    '$(0.300\\pm0.005)~\\times10^{23}$',
    '$(1.000\\pm0.100)~\\times10^{23}$']

    >>> latex_range_sci(x, x - err, x + err, ndigits = 2, base_pow=21)
    ['$1.00_{-0.10}^{+0.10}~\\times10^{21}$',
    '$30.00_{-0.50}^{+0.50}~\\times10^{21}$',
    '$100.00_{-10.00}^{+10.00}~\\times10^{21}$']
    """    
    n = len(best)
    x = best
    xmin = min
    xmax = max
    delta_min = x - xmin
    delta_max = xmax - x
    if base_pow is None:
        pow = np.trunc(np.log10(np.abs(xmin))) + 1
    else:
        pow = np.repeat(base_pow, len(xmin))

    if ndigits == "auto":
        err_min = np.min([xmax - x, x - xmin], axis=0)
        ndigits = calc_signif_digits(err_min / (10.0**pow))

    x = x / (10.0**pow)
    xmin = xmin / (10.0**pow)
    xmax = xmax / (10.0**pow)
    lo = round_vec(x - xmin, ndigits)
    up = round_vec(xmax - x, ndigits)

    if isinstance(ndigits, int) or isinstance(ndigits, float):
        check_ndigits(np.concatenate([lo,up]), ndigits)

    if np.all(lo == up) and symmetry:
        out_x = format_number(x, ndigits)
        out_pm = format_number(lo, ndigits)
        pow = pow.astype(int).astype(str)
        out = [
            "$(" + x_i + "\\pm" + x_i_err + ")~\\times10^{" + p + "}$"
            for x_i, x_i_err, p in zip(out_x, out_pm, pow)
        ]
    else:
        out_x = format_number(x, ndigits)
        out_lo = format_number(lo, ndigits)
        out_up = format_number(up, ndigits)
        pow = pow.astype(int).astype(str)
        out = ["$" + x_i + "_{-" + x_i_lo + "}^{+" + x_i_up + "}~\\times10^{" + p +  "}$"
            for x_i, x_i_lo, x_i_up, p in zip(out_x, out_lo, out_up, pow)
        ]
    return out