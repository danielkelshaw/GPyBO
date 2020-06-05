# GPyBO

[![Build Status](https://travis-ci.org/danielkelshaw/GPyBO.svg?branch=master)](https://travis-ci.org/danielkelshaw/GPyBO)

Gaussian Processes | Bayesian Optimisation

- [x] MIT License
- [x] Python 3.6+

### **Gaussian Process Regression in 10 Lines:**

```python
import numpy as np
from gpybo import GP, SquaredExponentialKernel

x = np.array([-4, -3, -2, -1, 1])
y = np.sin(x)

gp = GP(SquaredExponentialKernel()) | (x, y)
gp.train(n_restarts=10)

mu, cov = gp(np.linspace(-5, 5, 100))
```

###### Made by Daniel Kelshaw
