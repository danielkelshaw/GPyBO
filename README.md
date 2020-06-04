# GPyBO

[![Build Status](https://travis-ci.org/danielkelshaw/GPyBO.svg?branch=master)](https://travis-ci.org/danielkelshaw/GPyBO)

Gaussian Processes | Bayesian Optimisation

- [x] MIT License
- [x] Python 3.6+

### **Gaussian Process Regression in 10 Lines:**

```python
import torch
from gpybo import GP, SquaredExponentialKernel

x = torch.tensor([-4, -3, -2, -1, 1], dtype=torch.float32)
y = torch.sin(x)

gp = GP(SquaredExponentialKernel()) | (x, y)
gp.train(n_restarts=10)

mu, cov = gp(torch.linspace(-5, 5, 100))
```

###### Made by Daniel Kelshaw
