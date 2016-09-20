from pymc import *

import gaussian_model
model = MCMC(gaussian_model)
model.sample(iter=500)
print(model.stats())
