import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from pymc3 import Model, Normal, HalfNormal
from pymc3 import find_MAP
from pymc3 import NUTS, sample
from pymc3 import traceplot
from pymc3 import summary
from pymc3 import Slice

#data generation
alpha, sigma = 1, 1
beta = [1, 2.5]
train_size = 400
test_size = 100
size = train_size+test_size

X1 = np.random.randn(size)
X2 = np.random.randn(size) * 0.2
Y = alpha + beta[0]*X1 + beta[1]*X2 + np.random.randn(size)*sigma

X1_test = X1[train_size:size]
X2_test = X2[train_size:size]
Y_test = Y[train_size:size]

X1 = X1[0:train_size]
X2 = X2[0:train_size]
Y = Y[0:train_size]

fig, axes = plt.subplots(1, 2, sharex=True, figsize=(10,4))
axes[0].scatter(X1, Y)
axes[1].scatter(X2, Y)
axes[0].set_ylabel('Y') 
axes[0].set_xlabel('X1') 
axes[1].set_xlabel('X2')


# Model definition
basic_model = Model()

with basic_model:

    # Priors for unknown model parameters
    alpha = Normal('alpha', mu=0, sd=10)
    beta = Normal('beta', mu=0, sd=10, shape=2)
    sigma = HalfNormal('sigma', sd=1)

    # Expected value of outcome
    mu = alpha + beta[0]*X1 + beta[1]*X2

    # Likelihood (sampling distribution) of observations
    Y_obs = Normal('Y_obs', mu=mu, sd=sigma, observed=Y)



# MAP Inference
map_estimate = find_MAP(model=basic_model)

Y_predict = [(map_estimate['alpha']+(map_estimate['beta'][0]*X1_test[i])+(map_estimate['beta'][1]*X2_test[i]) ) for i in range(test_size)]
print "MAP SLE: "+str(sum([(Y_test[i] - Y_predict[i])**2 for i in range(test_size)]))



# MCMC Inference

with basic_model:

    # obtain starting values via MAP
    start = find_MAP(fmin=optimize.fmin_powell)

    # instantiate sampler
    step = Slice(vars=[sigma])

    # draw 2000 posterior samples
    #trace = sample(2000, start=start)

    # draw 5000 posterior samples
    trace = sample(5000, step=step, start=start)

#print trace['alpha'][-5:]

i_alpha = np.mean(trace['alpha'])
i_beta = trace['beta'].mean(0)

Y_predict = [(i_alpha+(i_beta[0]*X1_test[i])+(i_beta[1]*X2_test[i]) ) for i in range(test_size)]
print "MCMC SLE: "+str(sum([(Y_test[i] - Y_predict[i])**2 for i in range(test_size)]))

#summary(trace);

#plt.figure(figsize=(7, 7))
#traceplot(trace);
#plt.tight_layout();
#plt.show();



















