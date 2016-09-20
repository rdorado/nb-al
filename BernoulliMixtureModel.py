import random
import math

def randomNormalizedVector(k):
  vec = [random.random() for x in range(k)]
  Z = sum(vec)
  return [x/Z for x in vec]
  
def p(point, mu, d):
  ret = 1
  for i in range(d):
    if point[i] == 1:
      ret*=mu[i]
    else:
      ret*=(1-mu[i]) 
  return ret

def logp(point, mu, d):
  ret = 0
  for i in range(d):
    if point[i] == 1:
      ret+=math.log(mu[i])
    else:
      ret*=(1-mu[i]) 
  return ret
 
k = 2
#points = [{0:1,1:1,5:1}, {0:1,1:1}, {0:1,4:1,5:1}, {3:1,4:1,5:1}, {0:1,1:1,2:1}, {0:1,1:1,5:1}, {4:1,5:1}, {3:1,4:1,5:1}]
points = [[1,1,0,0,0,1], [0,0,0,0,0,1], [1,1,0,0,1,1], [0,0,0,1,1,1], [1,1,0,0,0,1], [0,0,0,0,0,1], [0,0,0,0,1,1], [0,0,0,1,1,1]]
n = len(points)
d = 6

lambdas = randomNormalizedVector(k)
#lambdas = [1/k for x in range(k)]
mu = [[random.random() for x in range(d)] for x in range(k)]

print "\nInit lamdas:"
print lambdas
print "\nInit mus:"
print mu
print "\n\n\n"

it=0
change=True

oldloglikelihood=0
while change:

  loglikelihood=0
  # Expectation
  Z = [[0.0 for x in range(k)] for x in range(n)]
  for i in range(n): 
    nsum=0
    for j in range(k):
      Z[i][j] = lambdas[j]*p(points[i], mu[j], d)
      loglikelihood+=math.log(Z[i][j]+0.0001)
      nsum+=Z[i][j]
    for j in range(k):
      Z[i][j]/=nsum

  if loglikelihood==oldloglikelihood:
    change=False
  else:
    oldloglikelihood=loglikelihood
#  print math.exp(loglikelihood)

  # Maximization
  sumZ = [[0.0 for x in range(d)] for x in range(k)]
  nlambdas = [0.0 for x in range(k)]
  nmu = [[0.0 for x in range(d)] for x in range(k)]
  for i in range(k):
    skmu = 0

    for j in range(n):
      skmu+=Z[j][i]
      for m in range(d):
        if points[j][m] == 1:
          nmu[i][m]+=Z[j][i]
      nlambdas[i] += Z[j][i] 
  
    for m in range(d):
      nmu[i][m]/=skmu 

    nlambdas[i]/=n

  #print nmu
  #print nlambdas
  mu = nmu
  lambdas=nlambdas

  ''' 
  # Calculate log-likelihood
  loglikelihood=0
  for i in range(n): 
    nsum=0
    for j in range(k):
      Z[i][j] = lambdas[j]*p(points[i], mu[j], d)  
  '''



print nmu
print nlambdas

