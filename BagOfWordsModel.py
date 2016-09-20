from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

from NaiveBayesActiveLearningModel import NaiveBayesActiveLearningModel

#Test libraries: to delete in future refactoring
import sys, getopt
from sklearn.datasets import fetch_20newsgroups

class BagOfWordsModel:

  def __init__(self):
    #self.modelType = modelType
    #self.size = size
    self.tokenizer = RegexpTokenizer(r'[a-z]+')
    self.SKIP_STOP_WORDS = True

    self.data=[]  
    self.vocabulary={}
    self.nterms=0
    self.documents=[]
    self.word_counts={}
    self.total_counts=0


  def getDocuments(self):
    return self.documents


  def getDocumentAsVector(self, index):
    return self.documents[index]


  def getVocabSize(self):
    return self.nterms
  

  def getCorpusAsVector(self):
    return self.word_counts


  def addDocument(self, document):
    splits = self.tokenizer.tokenize(document)

    if self.SKIP_STOP_WORDS:
       splits = [word for word in splits if word not in stopwords.words('english')]

    fd = {}
    for word in splits:
      try:
        id_word=self.vocabulary[word]
      except KeyError:
        self.vocabulary[word]=self.nterms
        id_word=self.nterms
        self.nterms+=1

      try:
        fd[id_word] = fd[id_word] + 1
      except KeyError:
        fd[id_word] = 1

      try:
        self.word_counts[id_word] = self.word_counts[id_word] + 1
      except KeyError:
        self.word_counts[id_word] = 1
      
    self.documents.append(fd)



def main(argv):

  # Data preprocessing
  try:
    opts, args = getopt.getopt(argv,"l:",["ifile=","ofile="])
  except getopt.GetoptError:
    # printUsage()
    sys.exit(2)

  categories = ['talk.politics.guns','soc.religion.christian','sci.electronics','rec.sport.baseball','comp.graphics']
  twenty_train = fetch_20newsgroups(categories=categories, subset='train', shuffle=True, random_state=42,remove=('headers', 'footers', 'quotes') )

  bow = BagOfWordsModel()

  #Initialization
   
  #categories = []
  #nb = NaiveBayesActiveLearningModel(bow,categories)

  #print length(twenty_train)

  # for doc in twenty_train.data:
  #  bow.addDocument(doc)

  bow.addDocument(twenty_train.data[12])
  bow.addDocument(twenty_train.data[9])
  #categories.append(twenty_train.target[9])
  #print twenty_train.target[9]

  print bow.getDocumentAsVector(0)
  print bow.getDocumentAsVector(1)
  print bow.getCorpusAsVector()
  #print bow.getDocumentAsVector(0)
  #nb.update() 

  #nb.predict()


  
#  while stopCriteria():
#    x = select()
#    retrain()
  
if __name__ == "__main__":
  main(sys.argv[1:])
