import torch.nn as nn


"""
At the moment the implementation of `GP` is rather messy and the inputs
are not consistent - this is leading to troubles with implementation of
Bayesian Optimisation.

This stage of work aims to focus on improving the quality of the GP
code such that it is not a blocker for the BO project.


// Stage One:
- [ ] Get everything working with 'hardcoded' shapes.
- [ ] Implement additional tests to make sure these work as intended.

// Stage Two:
- [ ] Add decorators to allow less constrained input (for users).
- [ ] Implement tests for decorators to ensure they work as intended.
- [ ] Add user-friendly interfaces such as __call__ and __repr__.
"""


class NewGP(nn.Module):
    pass
