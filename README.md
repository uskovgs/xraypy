# xraypy

Useful functions

<!-- badges: start -->
[![Lifecycle: experimental](https://img.shields.io/badge/lifecycle-experimental-orange.svg)](https://lifecycle.r-lib.org/articles/stages.html#experimental)
<!-- badges: end -->

## Installation

```bash
python -m pip install git+https://github.com/uskovgs/xraypy.git
```

## Usage

This module is designed for correct rounding of numbers.

```{python}
import numpy as np
import pandas as pd
from xraypy.latex_table import latex_range

B=3
x = np.random.uniform(1, 10, size=B)
err_lo = np.random.uniform(0, 1, size=B)
err_up = np.random.uniform(0, 1, size=B)
xmin = x - err_lo
xmax = x + err_up
```

Non symmetric errors:
```{python}
pd.DataFrame({
    "x" : x,
    "xmin" : xmin,
    "xmax" : xmax,
    "latex": latex_range(x, xmin, xmax)
})
```

```
          x      xmin       xmax                latex
0  1.821941  1.367433   2.155806  $1.8_{-0.5}^{+0.3}$
1  6.652371  5.905360   6.898622  $6.7_{-0.7}^{+0.2}$
2  9.423416  8.675003  10.069855  $9.4_{-0.7}^{+0.6}$
```
Symemtric errors:
```{python}
xmax = x + err_lo
pd.DataFrame({
    "x" : x,
    "xmin" : xmin,
    "xmax" : xmax,
    "latex": latex_range(x, xmin, xmax)
})
```

```
          x      xmin       xmax        latex
0  1.821941  1.367433   2.276448  $1.8\pm0.5$
1  6.652371  5.905360   7.399382  $6.7\pm0.7$
2  9.423416  8.675003  10.171828  $9.4\pm0.7$
```