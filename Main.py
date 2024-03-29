from BagOfWordsModel import BagOfWordsModel
import sys, getopt
from sklearn.datasets import fetch_20newsgroups
from BernoulliMixtureSparseModel import clusterBeurnoulliMixtureSparseModel

def main(argv):

  # Data preprocessing
  try:
    opts, args = getopt.getopt(argv,"l:",["ifile=","ofile="])
  except getopt.GetoptError:
    # printUsage()
    sys.exit(2)

  #categories = ['talk.politics.guns','soc.religion.christian','sci.electronics','rec.sport.baseball','comp.graphics']
  categories = ['rec.sport.baseball','comp.graphics']
  twenty_train = fetch_20newsgroups(categories=categories, subset='train', shuffle=True, random_state=42,remove=('headers', 'footers', 'quotes') )

  bow = BagOfWordsModel()

  #Initialization
   
  categories = []
  #nb = NaiveBayesActiveLearningModel(bow,categories)

  #print length(twenty_train)

  # for doc in twenty_train.data:
  #  bow.addDocument(doc)
  
  selected_docs = range(8)
  for indx in range(len(selected_docs)):
    bow.addDocument(twenty_train.data[indx])
    categories.append(twenty_train.target[indx])
    
  #bow.addDocument(twenty_train.data[9])
  #categories.append(twenty_train.target[9])

  #for i in range(len(selected_docs)):
  #  print "************\n Doc "+str(i)+": \n"
  #  print bow.getDocumentAsVector(i)
  #  print "Category: "+str(categories[i])

  #print "************\n Word Model: \n"
  #print bow.getCorpusAsVector()
  #print bow.getDocumentAsVector(0)
  #nb.update() 
  
  print "************\n Categories: \n"
  print categories
  points = bow.getDocuments()
  k = 3
  #points = [{0:1,1:1,2:1}, {2:1,3:1,4:1,5:1}, {0:1,1:1,2:1,5:1}, {2:1,3:1,4:1,5:1}, {0:1,1:1,2:1,5:1}, {2:1,3:1,4:1,5:1}, {0:1,1:1,2:1}, {2:1,3:1,4:1,5:1}]
  #points = [[1,1,1,0,0,0], [0,0,1,1,1,1], [1,1,1,0,0,1], [0,0,1,1,1,1], [1,1,1,0,0,1], [0,0,1,1,1,1], [1,1,1,0,0,0], [0,0,1,1,1,1]]  
  d = bow.getVocabSize();
  print d  

  clusterBeurnoulliMixtureSparseModel(k,points,d);

  #nb.predict()


  
#  while stopCriteria():
#    x = select()
#    retrain()
  
if __name__ == "__main__":
  main(sys.argv[1:])
