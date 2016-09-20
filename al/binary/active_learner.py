from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer


categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)


count_vect = CountVectorizer(binary='true')
X_train_counts = count_vect.fit_transform(twenty_train.data)
print X_train_counts
#twenty_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)
