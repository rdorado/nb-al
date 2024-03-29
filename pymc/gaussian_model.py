from pymc import *
import numpy as np

mu, sigma = 100, 40 # mean and standard deviation
data = np.random.normal(mu, sigma, 1000)
#data = map(float, open('data', 'r').readlines())
#print data

# Example 1
'''mean = Uniform('mean', lower=min(data), upper=max(data))
precision = Uniform('precision', lower=0.0001, upper=1.0)
process = Normal('process', mu=mean, tau=precision, value=data, observed=True)
'''

# Example 2
mean = Uniform('mean', lower=min(data), upper=max(data))
std_dev = Uniform('std_dev', lower=0, upper=50)

@deterministic(plot=False)
def precision(std_dev=std_dev):
  return 1.0/(std_dev*std_dev)

process = Normal('process', mu=mean, tau=precision, value=data, observed=True)
