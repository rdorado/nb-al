from pymc import *
import numpy as np
import matplotlib.pyplot as plt

#create the data
mu1, mu2, sigma = 100, 400, 40 # mean and standard deviation
data1 = np.random.normal(mu1, sigma, 1000)
data2 = np.random.normal(mu2, sigma, 1000)
data = np.append(data1, data2)
np.random.shuffle(data)

# the histogram of the data
n, bins, patches = plt.hist(data, 50, normed=1, facecolor='green', alpha=0.75)

plt.show()


theta = Uniform("theta", lower=0, upper=1)
bern = Bernoulli("bern", p=theta, size=len(data))

mean1 = Uniform('mean1', lower=min(data), upper=max(data))
mean2 = Uniform('mean2', lower=min(data), upper=max(data))
std_dev = Uniform('std_dev', lower=0, upper=50)

@deterministic(plot=False)
def mean(bern=bern, mean1=mean1, mean2=mean2):
    return bern * mean1 + (1 - bern) * mean2

@deterministic(plot=False)
def precision(std_dev=std_dev):
    return 1.0 / (std_dev * std_dev)

process = Normal('process', mu=mean, tau=precision, value=data, observed=True)

model = pymc.Model([process, precision, mean, std_dev, mean1, mean2, theta, bern])

mcmc = pymc.MCMC(model)
mcmc.sample(50000, 20000)

print "mean1: ",np.mean(mcmc.trace('mean1')[:])






