from pymc import *
import numpy as np

mu, sigma = 100, 40 # mean and standard deviation
data = np.random.normal(mu, sigma, 1000)
#data = map(float, open('data', 'r').readlines())
#print data

mean = Uniform('mean', lower=min(data), upper=max(data))
precision = Uniform('precision', lower=0.0001, upper=1.0)
process = Normal('process', mu=mean, tau=precision, value=data, observed=True)


