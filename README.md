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
from xraypy.latex_table import format_number,calc_n_digits,get_first_digit
import numpy as np

numbers = np.array([0.1234,0.4768,0.8598])
errors = np.array([0.198,0.0746,0.02577691])

format_number(numbers, calc_n_digits(errors))
```

['0.1', '0.48', '0.86']