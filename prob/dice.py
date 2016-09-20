import numpy as np
import pymc as pm

NUM_DRAWS = 100
NUM_SAMPLES = 50
TRUE_PROBS = [0.166, 0.166, 0.166, 0.166, 0.166, 0.166]
MIX_PROB = 0.3


def generate_data():
    data = []
    for i in range(NUM_SAMPLES):
        x = np.random.multinomial(NUM_DRAWS, TRUE_PROPS)
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

props = pm.Dirichlet("props", theta=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],)
draws = pm.Multinomial("draws", value=data, n=NUM_DRAWS, p=props, observed=True)

mcmc = pm.MCMC([props, draws])
mcmc.sample(iter=10000, burn=100) #, thin=100

summarize(mcmc, "props")

# mcmc.sample(iter=1000, burn=100, thin=1)
print mcmc.trace("props")[-200:][0]

