import pymc as pc  
import numpy as np 
import pymc.Matplot as pt   
from scipy.stats import bernoulli  
import random as rn  


def wordDict(collection):  
  word_id  = {}  
  idCounter = 0  
  for d in collection:  
    for w in d:  
      if (w not in word_id):  
        word_id[w] = idCounter  
        idCounter+=1  
  return word_id  
   
def toNpArray(word_id, collection):  
  ds = []  
  for d in collection:  
    ws = []  
    for w in d:  
      ws.append(word_id.get(w,0))  
    ds.append(ws)  
  return np.array(ds) 


docs = [["sepak","bola","sepak","bola","bola","bola","sepak"],  
         ["uang","ekonomi","uang","uang","uang","ekonomi","ekonomi"],  
         ["sepak","bola","sepak","bola","sepak","sepak"],  
         ["ekonomi","ekonomi","uang","uang"],  
         ["sepak","uang","ekonomi"],  
         ["komputer","komputer","teknologi","teknologi","komputer","teknologi"],  
         ["teknologi","komputer","teknologi"]]  

word_dict = wordDict(docs)  
collection = toNpArray(word_dict,docs)  

K = 3
V = len(word_dict)  
D = len(collection) 
alpha = np.ones(K)
beta = np.ones(V) 
Nd = [len(doc) for doc in collection]


#topic distribution per-document  
theta = pc.Container([pc.CompletedDirichlet("theta_%s" % i, pc.Dirichlet("ptheta_%s"%i, theta=alpha)) for i in range(D)]) 

#word distribution per-topic  
phi = pc.Container([pc.CompletedDirichlet("phi_%s" % j, pc.Dirichlet("pphi_%s" % j, theta=beta)) for j in range(K)]) 

#Please note that this is the tricky part :)  
z = pc.Container([pc.Categorical("z_%i" % d, p = theta[d], size = Nd[d], value = np.random.randint(K, size=Nd[d])) for d in range(D)])  

#word generated from phi, given a topic z  
w = pc.Container([pc.Categorical("w_%i_%i" % (d,i), p = pc.Lambda("phi_z_%i_%i" % (d,i), lambda z=z[d][i], phi=phi : phi[z]), value=collection[d][i], observed=True) for d in range(D) for i in range(Nd[d])]) 

model = pc.Model([theta, phi, z, w])  
mcmc = pc.MCMC(model)  
mcmc.sample(iter=5000, burn=1000)  
   
   
#show the topic assignment for each word, using the last trace  
for d in range(D):  
    print(mcmc.trace('z_%i'%d)[3999])








