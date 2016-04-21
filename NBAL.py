




class NaiveBayesActiveLearning:

  def __init__(self, modelType, size):
    self.modelType = modelType
    self.size = size
    self.data = []  



  def addDocument(self, document, category)

    for line in lines:
      line = line.lower()
      splits = tokenizer.tokenize(line)
      filtered_words = [word for word in splits if word not in stopwords.words('english')]
      filtered_words = [word for word in filtered_words if len(word) > 2]
      filtered_words = [word for word in filtered_words if word not in ["edu","com","subject","writes","mil", "subject"]]
      for word in filtered_words:
           
         try:
           id_word = vocab[word]
           count[id_word] += 1
         except:
           pass
    counts.append(count)
    iddoc += 1 
    #ncat = len( set(categories) )
