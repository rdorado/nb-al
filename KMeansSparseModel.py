import random
import sys
import copy

def distance(point1, point2):
  dist = 0
  for key in set(point1.keys()+point2.keys()):
    try:
      dist+=((point1[key]-point2[key])**2)
    except:
      try:
        dist+=(point1[key]**2) 
      except: 
        dist+=(point2[key]**2) 
  return dist

def clusterKMeansSparseModel(k, points, d):
  #k = 2
  #points = [{0:1,1:1,5:1}, {0:1,1:1}, {0:1,4:1,5:1}, {3:1,4:1,5:1}, {0:1,1:1,2:1}, {0:1,1:1,5:1}, {4:1,5:1}, {3:1,4:1,5:1}]
  #d = 6

  n = len(points)
  centroids = [copy.copy(points[int((n-1)*(float(x)/k))]) for x in range(k)]
  assignments = [int(random.random()*k) for x in range(n)]
  distances = [0.0 for x in range(n)]

  changed = True
  while changed:
    changed = False

    ## calculate new assignments based on centroids:
    for i in range(n):
      distances[i] = sys.maxint
      for j in range(k):
        dist = distance(points[i],centroids[j]) 
        if distances[i] > dist:
          best=j
          distances[i] = dist        

      if assignments[i] != best:
        assignments[i] = best
        changed = True

    #print assignments

    ## calculate new centroids
    temp = [{} for x in range(k)]
    npoints = [0.0 for x in range(k)]
    for i in range(n):
      for j in points[i].keys():
        try:
          temp[assignments[i]][j]+=points[i][j]
        except:
          temp[assignments[i]][j]=points[i][j]
      npoints[assignments[i]]+=1.0
 
    for i in range(k):
      for j in range(d):
        try:
          centroids[i][j]=temp[i][j]/npoints[i]
        except:
          centroids[i][j]=0

  
  print assignments
  print centroids 
  print distances


