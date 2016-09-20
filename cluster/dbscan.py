# A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise
# Martin Ester, Hans-Peter Kriegel, Jorg Sander, Xiaowei Xu
# dbscan: density based spatial clustering of applications with noise
import sys
import numpy as np
import math
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix
from nltk.corpus import stopwords
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from collections import defaultdict
import time



UNCLASSIFIED = False
NOISE = None



def get_max(assigns):
  d = defaultdict(int)
  for i in assigns:
    d[i] += 1
  return max(d.iteritems(), key=lambda x: x[1])


def add_to_centroid(centroid, point):
  for i in point.indices:
    if i not in centroid:
      centroid.append(i) 
  return centroid

   

def centroid_similarity(centroid, point):
  resp = 0
  for i in point.indices:
    if i in centroid:
      resp+=1 
  return resp


def _dist(p,q):
   return len(set(p.indices) ^ set(q.indices))
   '''
   for i, j in enumerate(p.indices):
     if j in q.indices:
        
        xi = p.data[i]
        xj = q.data[np.where(q.indices==j)[0][0]]
        print i," ",j," ",xi," ",xj
        #resp+=(xi-xj)**2
        resp+=abs(xi-xj) 
   #return math.sqrt(resp)
   return resp
   '''



def _eps_neighborhood(p,q,eps):
   dist = _dist(p,q)
   return dist < eps

# ******************************************************
#    return all points within p's eps-neighborhood (including p)
# ******************************************
def _region_query(m, point_id, eps):
    n_points = m.shape[0]
    seeds = []
    for i in range(0, n_points):
        if _eps_neighborhood(m.getrow(point_id), m.getrow(i), eps):
            seeds.append(i)
    return seeds

def _region_query_matdist(m, point_id, eps, matdist):
    n_points = m.shape[0]
    seeds = []
    for i in range(0, n_points):
        if matdist[point_id][i] < eps:
            seeds.append(i)
    return seeds


def _expand_cluster(m, classifications, point_id, cluster_id, eps, min_points):
    seeds = _region_query(m, point_id, eps)
    if len(seeds) < min_points:
        classifications[point_id] = NOISE
        return False
    else:
        classifications[point_id] = cluster_id
        for seed_id in seeds:
            classifications[seed_id] = cluster_id
            
        while len(seeds) > 0:
            current_point = seeds[0]
            results = _region_query(m, current_point, eps)
            if len(results) >= min_points:
                for i in range(0, len(results)):
                    result_point = results[i]
                    if classifications[result_point] == UNCLASSIFIED or \
                       classifications[result_point] == NOISE:
                        if classifications[result_point] == UNCLASSIFIED:
                            seeds.append(result_point)
                        classifications[result_point] = cluster_id
            seeds = seeds[1:]
        return True

def _expand_cluster_matdist(m, classifications, point_id, cluster_id, eps, min_points, matdist):
    seeds = _region_query_matdist(m, point_id, eps, matdist)
    if len(seeds) < min_points:
        classifications[point_id] = NOISE
        return False
    else:
        classifications[point_id] = cluster_id
        for seed_id in seeds:
            classifications[seed_id] = cluster_id
            
        while len(seeds) > 0:
            current_point = seeds[0]
            results = _region_query(m, current_point, eps)
            if len(results) >= min_points:
                for i in range(0, len(results)):
                    result_point = results[i]
                    if classifications[result_point] == UNCLASSIFIED or \
                       classifications[result_point] == NOISE:
                        if classifications[result_point] == UNCLASSIFIED:
                            seeds.append(result_point)
                        classifications[result_point] = cluster_id
            seeds = seeds[1:]
        return True

        
def dbscan(m, eps, min_points):
    """Implementation of Density Based Spatial Clustering of Applications with Noise
    See https://en.wikipedia.org/wiki/DBSCAN
    
    scikit-learn probably has a better implementation
    
    Uses Euclidean Distance as the measure
    
    Inputs:
    m - A matrix whose columns are feature vectors
    eps - Maximum distance two points can be to be regionally related
    min_points - The minimum number of points to make a cluster
    
    Outputs:
    An array with either a cluster id number or dbscan.NOISE (None) for each
    column vector in m.
    """no movie
    cluster_id = 1
    n_points = m.shape[0]
    classifications = [UNCLASSIFIED] * n_points
    for point_id in range(0, n_points):
        if classifications[point_id] == UNCLASSIFIED:
            if _expand_cluster(m, classifications, point_id, cluster_id, eps, min_points):
                cluster_id = cluster_id + 1
    return classifications


def dbscan_matdist(m, eps, min_points, matdist):
    """Implementation of Density Based Spatial Clustering of Applications with Noise
    See https://en.wikipedia.org/wiki/DBSCAN
    
    scikit-learn probably has a better implementation
    
    Uses Euclidean Distance as the measure
    
    Inputs:
    m - A matrix whose columns are feature vectors
    eps - Maximum distance two points can be to be regionally related
    min_points - The minimum number of points to make a cluster
    
    Outputs:
    An array with either a cluster id number or dbscan.NOISE (None) for each
    column vector in m.
    """no movie
    cluster_id = 1
    n_points = m.shape[0]
    classifications = [UNCLASSIFIED] * n_points
    for point_id in range(0, n_points):
        if classifications[point_id] == UNCLASSIFIED:
            if _expand_cluster_matdist(m, classifications, point_id, cluster_id, eps, min_points, matdist):
                cluster_id = cluster_id + 1
    return classifications


# **************************************************
#  Calculate the distance matrix between all points
# **************************************************
def calculate_matdist(m):
  n_points = m.shape[0]
  resp = [[0]*n_points for i in range(0, n_points)]
  for i in range(0, n_points-1): 
    p = m.getrow(i)     
    for j in range(i+1, n_points):
      resp[i][j] = resp[j][i] = _dist(p, m.getrow(j))
  return [resp, mins]

# **************************************************
#  Calculate the distance matrix between all points and also the minimum distance to next point for point i
# **************************************************
def calculate_matdist_with_mins(m):
  n_points = m.shape[0]
  resp = [[0]*n_points for i in range(0, n_points)]
  mins = []
  for i in range(0, n_points-1): 
    p = m.getrow(i)     
    minl = sys.maxint 
    for j in range(i+1, n_points):
      resp[i][j] = resp[j][i] = _dist(p, m.getrow(j))
      if minl > resp[i][j]: minl = resp[i][j]
    mins.append(minl)
  return [resp, mins]


# **************************************************
#    Extract the minimum distances for each point i 
# **************************************************
def get_mins(matdist):
   resp = []
   for i in range(len(matdist)):
     resp.append( min(matdist[i]) )
   print resp


def test_dbscan_2():

def test_dbscan():


    start = time.time()    no movie

    #  Loading training data
    print "Loading data..."
    count_vect = CountVectorizer(stop_words=stopwords.words('english'),binary=True)
    categories = ['rec.sport.baseball','comp.graphics']
    
    data = []
    for cat in categories:
      data = data + fetch_20newsgroups(categories=categories, subset='train', shuffle=True, random_state=42,remove=('headers', 'footers', 'quotes') ).data[:40]
    
    #data = [twenty_train.data[i] for i in [0,1,2,3,4,5,6,7,8,9]]
    #data = twenty_train.data
    m = count_vect.fit_transform(data)
    print "Data loaded..."

    #print m
    '''
    print X_train_counts.shape    
    n = X_train_counts.shape[0]
    data = []
    for i in range(n):
      for j in range(n):
        if i == j: continue
        data.append(  _dist(X_train_counts.geno movietrow(i),X_train_counts.getrow(j)) )

    print max(data)
    print min(data)   

    #''

    #col = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
    #row = np.array([0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6])
    #data = np.array([10, 12, 8, 37, 39, 36, 100, 11, 8, 10, 40, 39, 41, 100])
    m = csr_matrix((data, (row, col)), shape=(4, 10))

    
    #''
    col = np.array([0, 1, 2, 3, 1, 4, 5, 4, 6, 7, 8, 1, 9, 5])
    row = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])
    data = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    m = csr_matrix((data, (row, col)), shape=(4, 10))
    #'''

    print "Loading data time: ",(time.time() no movie- start)," ms"
    start = time.time()

    print "Calculating distances..."
    matdist, mins = calculate_matdist_with_mins(m)
    #mins = get_mins(matdist)
    #print matdist
    eps = np.median(mins)
    min_points = 2
    #print eps

    print "Distance matrix time: ",(time.time() - start)," ms"
    start = time.time()


    print "Performing clustering..."
    assigns = dbscan_matdist(m, eps, min_points, matdist)
    print "DB scan time : ",(time.time() - start)
    start = time.time()


    print "Finding next point..."
    clusterid = get_max(assigns)[0]
    centroid = []no movie

    print "Finfing best point..."
    for i, item in enumerate(assigns):      
      if item == clusterid:
        centroid = add_to_centroid(centroid, m.getrow(i))

    maxsim = 0
    maxid = -1
    for i, item in enumerate(assigns):      
      if item == clusterid:
        sim = centroid_similarity(centroid, m.getrow(i))
        if sim > maxsim: 
          maxsim = sim
          maxid = i

    print "Searching best point : ",(time.time() - start)
    print "Selected: ",maxid
    #'''

#test_dbscan()


start = time.time()    

#  Loading training data
print "Loading data..."
count_vect = CountVectorizer(stop_words=stopwords.words('english'),binary=True)
categories = ['rec.sport.baseball','comp.graphics']
    
data = []
for cat in categories:
   data = data + fetch_20newsgroups(categories=categories, subset='train', shuffle=True, random_state=42,remove=('headers', 'footers', 'quotes') ).data[:40]
    
m = count_vect.fit_transform(data)
print "Data loaded. Time spent:",(time.time() - start)," ms"
start = time.time()


print "Calculating distances..."
matdist, mins = calculate_matdist_with_mins(m)
print "Distance matrix time: ",(time.time() - start)," ms"
start = time.time()


