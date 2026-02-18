import numpy as np
import pandas as pd
import warnings







def round_vec(arr: np.ndarray, decimals) -> np.ndarray:
    if np.isscalar(decimals):
        decimals = np.full_like(arr, int(decimals), dtype=int)
    else:
        decimals = np.array(decimals, dtype=int)
        if arr.shape != decimals.shape:
            raise ValueError("arr and decimals must have the same shape")
    return np.round(arr * 10**decimals) / 10**decimals


def round_preserve(x, decimals):
    if np.isscalar(x):
        return round(x, int(decimals))
    if isinstance(x, pd.Series):
        rounded = round_vec(x.to_numpy(), decimals)
        return pd.Series(rounded, index=x.index, name=x.name)
    if isinstance(x, np.ndarray):
        return round_vec(x, decimals)
    if isinstance(x, (list, tuple)):
        arr = np.array(x)
        result = round_vec(arr, decimals).tolist()
        return tuple(result) if isinstance(x, tuple) else result
    try:
        return round_vec(np.array(x), decimals)
    except Exception:
        raise TypeError(f"Unsupported type: {type(x)}")


def check_ndigits(x, ndigits=0):
    rounded_x = round_vec(np.array(x), ndigits)
    if np.any(rounded_x == 0):
        warnings.warn("Is the argument `ndigits` too small?", UserWarning)


def get_first_digit(x):
    arr = np.array(x)
    tmp = np.round(np.abs(np.log10(np.abs(arr))), 0)
    first = np.abs((10 ** tmp) * arr)
    return np.floor(first).astype(int)


def get_n_signif_digits(x):
    arr = np.array(x)
    fd = get_first_digit(arr)
    n = np.abs(np.ceil(-np.log10(np.abs(arr))))
    n = np.where(fd == 1, n + 1, n)
    n[np.abs(arr) > 2] = 0
    return n.astype(int)


def calc_signif_digits(x):
    arr = np.array(x)
    n = get_n_signif_digits(arr)
    rounded = round_vec(arr, n)
    adjust = get_n_signif_digits(rounded) < n
    return np.where(adjust, n - 1, n)


def format_number(x, n_signif_digits=0):
    arr = np.atleast_1d(x)
    if np.isscalar(n_signif_digits):
        digs = np.full(arr.shape, int(n_signif_digits), dtype=int)
    else:
        digs = np.array(n_signif_digits, dtype=int)
    return np.array([f"%.{d}f" % round(val, d) for val, d in zip(arr, digs)], dtype=str)


def _as_numpy(x):
    if isinstance(x, pd.Series):
        return x.to_numpy()
    if np.isscalar(x):
        return np.array([x])
    return np.array(x)


def _restore(result_arr, orig_type, template=None):
    converters = {
        pd.Series: lambda arr: pd.Series(arr, index=template.index, name=template.name),
        np.ndarray: lambda arr: arr,
        list:    lambda arr: arr.tolist(),
        tuple:   lambda arr: tuple(arr.tolist()),
        "scalar": lambda arr: arr[0]
    }
    if orig_type in (int, float):
        return result_arr[0]
    key = orig_type if orig_type in converters else (
        pd.Series if template is not None else (
            list if isinstance(template, list) else (
                tuple if isinstance(template, tuple) else np.ndarray
            )
        )
    )
    return converters[key](result_arr)


def latex_range(best, lo, up, ndigits='auto'):
    orig_type = type(best)
    template = best if isinstance(best, pd.Series) else None
    best_a = _as_numpy(best)
    lo_a = _as_numpy(lo)
    up_a = _as_numpy(up)

    if ndigits == 'auto':
        errs = np.minimum(up_a - best_a, best_a - lo_a)
        nd = calc_signif_digits(errs)
    else:
        nd = ndigits

    lo_diff = round_vec(best_a - lo_a, nd)
    hi_diff = round_vec(up_a - best_a, nd)

    if np.isscalar(nd):
        check_ndigits(np.concatenate([lo_diff, hi_diff]), nd)

    out = []
    if np.all(lo_diff == hi_diff):
        xs = format_number(best_a, nd)
        es = format_number(lo_diff, nd)
        out = [f"${x}\pm{e}$" for x, e in zip(xs, es)]
    else:
        xs   = format_number(best_a, nd)
        los  = format_number(lo_diff, nd)
        his  = format_number(hi_diff, nd)
        out = [f"${x}_{{-{l}}}^{{+{h}}}$" for x, l, h in zip(xs, los, his)]

    result = np.array(out, dtype=str)
    return _restore(result, orig_type, template)


def latex_range_sci(best, lo, up, ndigits='auto', symmetry=False, base_pow=None):
    orig_type = type(best)
    template = best if isinstance(best, pd.Series) else None
    best_a = _as_numpy(best)
    lo_a   = _as_numpy(lo)
    up_a   = _as_numpy(up)

    if base_pow is None:
        pow_a = np.trunc(np.log10(np.abs(lo_a))) + 1
    else:
        pow_a = np.full_like(best_a, base_pow, dtype=int)

    if ndigits == 'auto':
        errs = np.minimum(up_a - best_a, best_a - lo_a) / (10**pow_a)
        nd = calc_signif_digits(errs)
    else:
        nd = ndigits

    x_s   = best_a / (10**pow_a)
    lo_s  = (best_a - lo_a) / (10**pow_a)
    hi_s  = (up_a - best_a) / (10**pow_a)

    lo_diff = round_vec(lo_s, nd)
    hi_diff = round_vec(hi_s, nd)

    if np.isscalar(nd):
        check_ndigits(np.concatenate([lo_diff, hi_diff]), nd)

    pow_str = pow_a.astype(int).astype(str)
    out = []
    if symmetry and np.all(lo_diff == hi_diff):
        xs = format_number(x_s, nd)
        es = format_number(lo_diff, nd)
        out = [f"$({x}\pm{e})~\times10^{{{p}}}$" for x, e, p in zip(xs, es, pow_str)]
    else:
        xs  = format_number(x_s, nd)
        los = format_number(lo_diff, nd)
        his = format_number(hi_diff, nd)
        out = [f"${x}_{{-{l}}}^{{+{h}}}~\times10^{{{p}}}$" for x, l, h, p in zip(xs, los, his, pow_str)]

    result = np.array(out, dtype=str)
    return _restore(result, orig_type, template)
