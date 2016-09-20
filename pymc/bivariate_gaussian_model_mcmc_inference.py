from pymc import *

import bivariate_gaussian_model
model = MCMC(bivariate_gaussian_model)
model.sample(iter=500)
print(model.stats())

import numpy
for p in ['mean', 'std_dev']:
    numpy.savetxt("%s.trace" % p, model.trace(p)[:])
