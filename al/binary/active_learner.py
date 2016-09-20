from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer


categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)


count_vect = CountVectorizer(binary='true')
csr = count_vect.fit_transform(twenty_train.data)

start = 0
counts = [{} for i in range(len(categories))]
for i, end in enumerate(csr.indptr[1:]):
    print i
    for j, val in zip(csr.indices[start:end], csr.data[start:end]):
      try:
        counts[twenty_train.target[i]][j] = counts[twenty_train.target[i]][j]+1
      except KeyError:
        counts[twenty_train.target[i]][j] = 1  
        #arr[i,j] = val
    start = end

#print type(X_train_counts)
#twenty_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)
