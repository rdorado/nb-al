'''
import numpy as np
import pymc3 as pm
from scipy import sparse as sp
from scipy.optimize import fmin_l_bfgs_b
from scipy.special import expit
from theano import sparse as S

#np.random.seed(0)

# properties of sparse design matrix (taken from the real data)
N = 1000  # number of samples
M = 5000    # number of dimensions
D = 0.002   # matrix density

# fake data
mu0, sd0 = 0.0, 1.0
w = np.random.normal(mu0, sd0, M)
X = sp.random(N, M, density=D, format='csr', data_rvs=np.ones)
y = np.random.binomial(1, expit(X.dot(w)), N)

# estimate memory usage
size = X.data.nbytes + X.indices.nbytes + X.indptr.nbytes + y.nbytes
print('{:.2f} MB of data'.format(size / 1024 ** 2))
'''

'''
# model definition
with pm.Model() as model:
    w = pm.Normal('w', mu0, sd=sd0, shape=M)

    p = pm.sigmoid(S.dot(X, w.reshape((-1, 1)))).flatten()
    pm.Bernoulli('y', p, observed=y)

    print(pm.find_MAP(vars=[w], fmin=fmin_l_bfgs_b))
'''




'''
#data = map(float, open('data', 'r').readlines())
 
theta = Uniform("theta", lower=0, upper=1)
bern = Bernoulli("bern", p=theta, size=len(data))
 
#mean1 = Uniform('mean1', lower=min(data), upper=max(data))
#mean2 = Uniform('mean2', lower=min(data), upper=max(data))
#std_dev = Uniform('std_dev', lower=0, upper=50)
 
#@deterministic(plot=False)
#def mean(bern=bern, mean1=mean1, mean2=mean2):
#    return bern * mean1 + (1 - ber) * mean2
 
#@deterministic(plot=False)
#def precision(std_dev=std_dev):
#    return 1.0 / (std_dev * std_dev)
 
#process = Normal('process', mu=mean, tau=precision, value=X, observed=True)
'''

'''
sigmas = pm.Normal('sigmas', mu=0.1, tau=1000, size=2)
centers = pm.Normal('centers', [0.3, 0.7], [1/(0.1)**2, 1/(0.1)**2], size=2)
alpha  = pm.Beta('alpha', alpha=2, beta=3)

category = pm.Container([pm.Categorical("category%i" % i, [alpha, 1 - alpha]) for i in range(nsamples)])

observations = pm.Container([pm.Normal('samples_model%i' % i, mu=centers[category[i]], tau=1/(sigmas[category[i]]**2),  value=samples[i], observed=True) for i in range(nsamples)])
model = pm.Model([observations, category, alpha, sigmas, centers])
#mcmc = pm.MCMC(model)
'''

'''
import numpy as np, math, matplotlib.pyplot as plt, pandas as pd

params = [(150, 10, 0.7),(200, 20, 0.3)]
no_samples = 1000

dist = [np.random.normal(mu, sigma, math.floor(mixing_prob * no_samples)) for (mu, sigma, mixing_prob) in params]
xs = np.array([item for sublist in dist for item in sublist])



import pymc as pm

p = pm.Uniform("p", 0, 1)
assignment = pm.Categorical("assignment", [p, 1 - p], size = xs.shape[0])

taus = 1.0 / pm.Uniform("stds", 0, 100, size = 2) **2
centers = pm.Normal("centers", [145, 210], [0.01, 0.02], size = 2)

@pm.deterministic
def center_i(assignment=assignment, centers=centers):
    return centers[assignment]

@pm.deterministic
def tau_i(assignment=assignment, taus=taus):
    return taus[assignment]

observations = pm.Normal("obs", center_i, tau_i, value=xs, observed=True)

model = pm.Model([p, assignment, observations, taus, centers])

mcmc = pm.MCMC(model)
mcmc.sample(40000, 10000)
mcmc.sample(10000)


center_trace = mcmc.trace("centers")[:]

centers = [['Center 1', x] for x in center_trace[:, 0]]
centers += [['Center 2', x] for x in center_trace[:, 1]]

print "Center 1: ",np.mean(center_trace[:, 0])
print "Center 2: ",np.mean(center_trace[:, 1])

df = pd.DataFrame(data = centers)
df.columns = ['Posterior', 'Value']
'''

'''
import pymc as pm
import numpy as np

mu, sigma = 100, 40 # mean and standard deviation
data = np.random.normal(mu, sigma, 1000)

mean = pm.Uniform('mean', lower=min(data), upper=max(data))
precision = pm.Uniform('precision', lower=0.0001, upper=1.0)
process = pm.Normal('process', mu=mean, tau=precision, value=data, observed=True)

model = pm.Model([process, precision, mean])
mcmc = pm.MCMC(model)
mcmc.sample(10000)

print np.mean(mcmc.trace("mean")[-100:])
'''


'''
import numpy as np

mean = [0, 0, 0]
cov = [[1, 0, 0], [0, 100, 0], [0, 0, 50]] 

import matplotlib.pyplot as plt
data = np.random.multivariate_normal(mean, cov, 5000).T

plt.plot(data[0], data[1], 'x')
plt.axis('equal')
plt.show()
'''

import numpy as np
import pymc as pm

NUM_DRAWS = 100
NUM_SAMPLES = 50
TRUE_PROBS = [[0.166, 0.166, 0.166, 0.166, 0.166, 0.166],[0.3, 0.2, 0.2, 0.1, 0.1, 0.1]]
MIX_PROB = 0.3


def generate_data():
    data = []
    for i in range(NUM_SAMPLES):
        p = np.random.random()
        if p < MIX_PROB:
          x = np.random.multinomial(NUM_DRAWS, TRUE_PROBS[0])
        else:
          x = np.random.multinomial(NUM_DRAWS, TRUE_PROBS[1])
        data.append(x)
    return data


def summarize(mcmc, field):
    results = mcmc.trace(field)[:]
    results = zip(*results)
    means = []
    for r in results:
        #m, v = stats.mean_and_sample_variance(r)
        m = np.mean(r)
        means.append(m)
    means.append(1.0 - sum(means))
    print
    print "---"
    print means




data = generate_data()

print data

theta = np.ones(6)
#props = pm.Dirichlet("props", theta=theta)
props = pm.Container([pm.Dirichlet("theta_%s" % i, theta=theta) for i in range(2)])

draws = pm.Multinomial("draws", value=data, n=NUM_DRAWS, p=props, observed=True)

mcmc = pm.MCMC([props, draws])
mcmc.sample(iter=10000, burn=100) #, thin=100

summarize(mcmc, "props")

# mcmc.sample(iter=1000, burn=100, thin=1)
print mcmc.trace("props")[-200:][0]

















