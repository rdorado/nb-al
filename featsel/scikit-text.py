from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from sklearn.feature_selection import VarianceThreshold

#print stopwords.words('english')

categories = ['rec.sport.baseball','comp.graphics']
twenty_train = fetch_20newsgroups(categories=categories, subset='train', shuffle=True, random_state=42,remove=('headers', 'footers', 'quotes') )

selected = [9, 11, 12, 13, 14, 15, 16, 17, 18, 19]
data = [twenty_train.data[i] for i in selected]

print [twenty_train.target[i] for i in selected]

#for i in range(len(data)):
#  print "Doc "+str(i+1)+":"
#  print data[i]+"\n"

#count_vect = CountVectorizer(stop_words="english")
#X_train_counts = count_vect.fit_transform(data)
#print count_vect.vocabulary_
#print "V size: "+str(len(count_vect.vocabulary_))+"\n\n"

data = ['red red red purple purple purple','blue blue purple green','red purple blue blue','purple purple blue blue blue','purple blue red blue','blue purple blue red']

count_vect = CountVectorizer(stop_words=stopwords.words('english')+['doesn','ve','dont'])
X_train_counts = count_vect.fit_transform(data)

for row in X_train_counts:
  for col in row:
    print "->"+str(col)+"\n"

#print X_train_counts
#print count_vect.vocabulary_
#print "V size: "+str(len(count_vect.vocabulary_))+"\n\n"
print "\n"
sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
print sel.fit_transform(X_train_counts)




