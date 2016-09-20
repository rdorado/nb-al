from bayespy.nodes import GaussianARD, Gamma, Categorical, Dirichlet, Beta, Mixture, Bernoulli
from bayespy.inference import VB
import bayespy.plot as bpplt
from bayespy.utils import random

import numpy as np
from scipy import special, optimize

import pprint

'''
data = np.random.normal(5, 10, size=(10,))


mu = GaussianARD(0, 1e-6)
tau = Gamma(1e-6, 1e-6)
y = GaussianARD(mu, tau, plates=(10,))

	
y.observe(data)

Q = VB(mu, tau, y)

Q.update(repeat=20)

bpplt.pyplot.subplot(2, 1, 1)
bpplt.pdf(mu, np.linspace(-10, 20, num=100), color='k', name=r'\mu')

bpplt.pyplot.subplot(2, 1, 2)
bpplt.pdf(tau, np.linspace(1e-6, 0.08, num=100), color='k', name=r'\tau')

bpplt.pyplot.tight_layout()
bpplt.pyplot.show()
'''

p0 = [0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1]
p1 = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9]
p2 = [0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

p = np.array([p0, p1, p2])
z = random.categorical([1/3, 1/3, 1/3], size=100)
x = random.bernoulli(p[z])

N = 100
D = 10
K = 3

R = Dirichlet(K*[1e-5],name='R')
Z = Categorical(R,plates=(N,1),name='Z')
P = Beta([0.5, 0.5],plates=(D,K),name='P')
X = Mixture(Z, Bernoulli, P)

Q = VB(Z, R, X, P)
P.initialize_from_random()
X.observe(x)

Q.update(repeat=1000)

#print(" P:")
#print( P.get_moments() )

#print(" R:")
#print( R.get_moments() )

print(" Z:")
print( Z.get_moments() )

print(" X:")
print( X.get_moments() )


bpplt.hinton(R)
#bpplt.hinton(P)
#bpplt.hinton(Z)

bpplt.pyplot.show()

#pp = pprint.PrettyPrinter(indent=4)

#pp.pprint(X)












