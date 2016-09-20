from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import mixture

import numpy as np
import pymc as pm

categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
twenty_test = fetch_20newsgroups(subset='test',categories=categories, shuffle=True, random_state=42)      

train_data = twenty_train.data
train_targets = twenty_train.target
test_data = twenty_test.data
test_targets = twenty_test.target


count_vect = CountVectorizer()
X_train = count_vect.fit_transform(train_data)



data = X_train.toarray()
K = 2 # number of topics
V = 4 # number of words
D = 3 # number of documents

alpha = np.ones(K)
beta = np.ones(V+1)

theta = pm.Container([pm.Dirichlet("theta_%s" % i, theta=alpha) for i in range(D)])
phi = pm.Container([pm.Dirichlet("phi_%s" % k, theta=beta) for k in range(K)])
Wd = [len(doc) for doc in data]

z = pm.Container([pm.Categorical('z_%i' % d, p = theta[d], size=Wd[d], value=np.random.randint(K, size=Wd[d])) for d in range(D)])
w = pm.Container([pm.Categorical("w_%i_%i" % (d,i), p = pm.Lambda('phi_z_%i_%i' % (d,i), lambda z=z[d][i], phi=phi: phi[z]), value=data[d][i], observed=True) for d in range(D) for i in range(Wd[d])])

model = pm.Model([theta, phi, z, w])
mcmc = pm.MCMC(model)
mcmc.sample(100)

#gmm = mixture.DPGMM(n_components=2, covariance_type='full')
#gmm.fit( X_train.toarray() )

#print gmm.means_ 


#clf = MultinomialNB().fit(X_train, train_targets)   

#X_new = count_vect.transform(test_data)
#predicted = clf.predict(X_new)
#print 'Accuracy: ',np.mean(predicted == test_targets) 
