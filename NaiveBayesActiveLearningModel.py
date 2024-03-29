import numpy as np
import math

class NaiveBayesActiveLearningModel:

  def __init__(self, languageModel, categories):
     self.languageModel=languageModel
     self.categories=categories
     self.sumscounts = [] 
     self.ncat=0

  def update(self):
     nterms=self.languageModel.nterms
     sumall = np.zeros(nterms)
     self.ncat = len( set(self.categories) )

     countcat = [0 for x in range(self.ncat)]
     for i in self.categories:
       countcat[i]+=1

     for i in range(self.ncat):
       self.sumscounts.append( np.zeros(nterms) )  

     logcats = [math.log(x/float(len(self.categories))) for x in countcat]
     sumcats = [0 for x in range(self.ncat)]



