from scipy.stats import distributions
import numpy as np
import pymc as pm

import matplotlib.pyplot as plt

# Sample parameters
nsamples = 1000
mu1_true = 0.3
mu2_true = 0.55
sig1_true = 0.08
sig2_true = 0.12
a_true = 0.4

# Samples generation
#np.random.seed(3)  # for repeatability
s1 = distributions.norm.rvs(mu1_true, sig1_true, size=round(a_true*nsamples))
s2 = distributions.norm.rvs(mu2_true, sig2_true, size=round((1-a_true)*nsamples))
samples = np.hstack([s1, s2])
bins = np.arange(-0.2, 1.2, 0.01)
data, _ = np.histogram(samples, bins=bins, density=True)
x_data = bins[:-1] + 0.5*(bins[1] - bins[0])

plt.plot(x_data, data, '-o');
plt.show()


# Least-squares histogram fit

#Model definition
import lmfit

peak1 = lmfit.models.GaussianModel(prefix='p1_')
peak2 = lmfit.models.GaussianModel(prefix='p2_')
model = peak1 + peak2

model.set_param_hint('p1_center', value=0.2, min=-1, max=2)
model.set_param_hint('p2_center', value=0.5, min=-1, max=2)
model.set_param_hint('p1_sigma', value=0.1, min=0.01, max=0.3)
model.set_param_hint('p2_sigma', value=0.1, min=0.01, max=0.3)
model.set_param_hint('p1_amplitude', value=1, min=0.0, max=1)
model.set_param_hint('p2_amplitude', expr='1 - p1_amplitude')
name = '2-gaussians'


# Fit

fit_res = model.fit(data, x=x_data, method='nelder')
print fit_res.fit_report()

fig, ax = plt.subplots()
x = x_data
ax.plot(x, model.eval(x=x, **fit_res.values), 'k', alpha=0.8)
plt.plot(x_data, data, 'o');
if  fit_res.model.components is not None:
    for component in fit_res.model.components:
        ax.plot(x, component.eval(x=x, **fit_res.values), '--k',
                alpha=0.8)
for param in ['p1_center', 'p2_center']:
    ax.axvline(fit_res.params[param].value, ls='--', color='r')
ax.axvline(mu1_true, color='k', alpha=0.5)
ax.axvline(mu2_true, color='k', alpha=0.5)

plt.show()


# Bayesian Inference: MCMC 
# Method 1
'''
sigmas = pm.Normal('sigmas', mu=0.1, tau=1000, size=2)
centers = pm.Normal('centers', [0.3, 0.7], [1/(0.1)**2, 1/(0.1)**2], size=2)

alpha  = pm.Beta('alpha', alpha=2, beta=3)
category = pm.Categorical("category", [alpha, 1 - alpha], size=nsamples)

@pm.deterministic
def mu(category=category, centers=centers):
    return centers[category]

@pm.deterministic
def tau(category=category, sigmas=sigmas):
    return 1/(sigmas[category]**2)

observations = pm.Normal('samples_model', mu=mu, tau=tau, value=samples, observed=True)

gen_model = pm.Normal('gen_model', mu=mu, tau=tau)  # generative model


t3 = pm.rbeta(alpha=6, beta=2, size=1e5)
t4 = pm.rbeta(alpha=2, beta=3, size=1e5)
plt.hist(t3, bins=bins, alpha=0.5);
plt.hist(t4, bins=bins, alpha=0.5);


t1 = pm.rnormal(0.3, 1/(0.1)**2, size=1e4)
t2 = pm.rnormal(0.55, 1/(0.1)**2, size=1e4)
plt.hist(t1, bins=bins, alpha=0.5)
plt.hist(t2, bins=bins, alpha=0.5);


model = pm.Model([observations, mu, tau, category, alpha, sigmas, centers])
mcmc = pm.MCMC(model)
mcmc.sample(100000, 30000)
pm.Matplot.plot(mcmc)
'''

# Method 2

sigmas = pm.Normal('sigmas', mu=0.1, tau=1000, size=2)
centers = pm.Normal('centers', [0.3, 0.7], [1/(0.1)**2, 1/(0.1)**2], size=2)
alpha  = pm.Beta('alpha', alpha=2, beta=3)
category = pm.Container([pm.Categorical("category%i" % i, [alpha, 1 - alpha]) for i in range(nsamples)])
observations = pm.Container([pm.Normal('samples_model%i' % i, 
                   mu=centers[category[i]], tau=1/(sigmas[category[i]]**2), 
                   value=samples[i], observed=True) for i in range(nsamples)])
model = pm.Model([observations, category, alpha, sigmas, centers])
mcmc = pm.MCMC(model)
# initialize in a good place to reduce the number of steps required
centers.value = [mu1_true, mu2_true]
# set a custom proposal for centers, since the default is bad
mcmc.use_step_method(pm.Metropolis, centers, proposal_sd=sig1_true/np.sqrt(nsamples))
# set a custom proposal for category, since the default is bad
for i in range(nsamples):
    mcmc.use_step_method(pm.DiscreteMetropolis, category[i], proposal_distribution='Prior')
mcmc.sample(100)  # beware sampling takes much longer now
# check the acceptance rates
print mcmc.step_method_dict[category[0]][0].ratio
print mcmc.step_method_dict[centers][0].ratio
print mcmc.step_method_dict[alpha][0].ratio

