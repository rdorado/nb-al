import collections, itertools
import nltk.classify.util, nltk.metrics
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews, stopwords
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.datasets import fetch_20newsgroups

import numpy as np
import re
import os, sys, io

#reload(sys)
#sys.setdefaultencoding('utf8')
 
def evaluate_classifier(featx, dataset, encod=""):
    if dataset=="movies":
      negids = movie_reviews.fileids('neg')
      posids = movie_reviews.fileids('pos')
      
      #print movie_reviews.raw(fileids=[negids[0]])
      negtexts = [preprocess(movie_reviews.raw(fileids=[f]),'text') for f in negids] 
      posfexts = [preprocess(movie_reviews.raw(fileids=[f]),'text') for f in posids]

      negfeats = [(featx(movie_reviews.words(fileids=[f])), 'neg') for f in negids]
      posfeats = [(featx(movie_reviews.words(fileids=[f])), 'pos') for f in posids]
    
      negfeats2 = [(preprocess(movie_reviews.raw(fileids=[f]),'dict'), 'neg') for f in negids]
      posfeats2 = [(preprocess(movie_reviews.raw(fileids=[f]),'dict'), 'pos') for f in posids]

      Nneg = len(negfeats)
      Npos = len(posfeats)
      negcutoff = Nneg*3/4
      poscutoff = Npos*3/4
    
      trainfeats = negfeats[:negcutoff] + posfeats[:poscutoff]
      testfeats = negfeats[negcutoff:] + posfeats[poscutoff:]

      trainfeats2 = negfeats2[:negcutoff] + posfeats2[:poscutoff]
      testfeats2 = negfeats2[negcutoff:] + posfeats2[poscutoff:]

      train_data = negtexts[:negcutoff] + posfexts[:poscutoff]
      train_targets = np.append(np.full_like(np.arange(negcutoff, dtype=np.int),0) , np.full_like(np.arange(poscutoff, dtype=np.int),1))
      test_data = negtexts[negcutoff:] + posfexts[poscutoff:]
      test_targets = np.append(np.full_like(np.arange(Nneg-negcutoff, dtype=np.int),0) , np.full_like(np.arange(Npos-poscutoff, dtype=np.int),1))

    elif dataset=="20newsgroups-5":   

      categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
      twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
      twenty_test = fetch_20newsgroups(subset='test',categories=categories, shuffle=True, random_state=42)      

      train_data = twenty_train.data
      train_targets = twenty_train.target
      test_data = twenty_test.data
      test_targets = twenty_test.target

      trainfeats = [(featx(preprocess(train_data[i],'words')), train_targets[i]) for i in range(len(train_data))]
      trainfeats2 = [(preprocess(train_data[i],'dict'), train_targets[i]) for i in range(len(train_data))]
      testfeats = [(featx(preprocess(test_data[i],'words')), test_targets[i]) for i in range(len(test_data))]
      testfeats2 = [(preprocess(test_data[i],'dict'), test_targets[i]) for i in range(len(test_data))]

    elif dataset=="20newsgroups":   

      twenty_train = fetch_20newsgroups(subset='train', shuffle=True, random_state=42)
      twenty_test = fetch_20newsgroups(subset='test', shuffle=True, random_state=42)      

      train_data = twenty_train.data
      train_targets = twenty_train.target
      test_data = twenty_test.data
      test_targets = twenty_test.target

      trainfeats = [(featx(preprocess(train_data[i],'words')), train_targets[i]) for i in range(len(train_data))]
      trainfeats2 = [(preprocess(train_data[i],'dict'), train_targets[i]) for i in range(len(train_data))]
      testfeats = [(featx(preprocess(test_data[i],'words')), test_targets[i]) for i in range(len(test_data))]
      testfeats2 = [(preprocess(test_data[i],'dict'), test_targets[i]) for i in range(len(test_data))]

    else:
      # Open a file
      path = dataset
      cat_dirs = os.listdir( path )
      
      print "Reading corpus from "+path  
      # This would print all the files and directories
      ncat = 0

      train_data = []
      train_targets = []
      test_data = []
      test_targets = []
  
      
      for category in cat_dirs:
        print "Reading category: "+category

        cat_files = os.listdir( path+"/"+category )

        temp_data = []
        temp_targets = []
        #encod = 'utf-8'

        for filename in cat_files:
          with io.open(path+"/"+category+"/"+filename, 'r', encoding=encod) as file:
            content = preprocess(file.read(),"text")
            temp_data.append(content)
         
          temp_targets.append(ncat)

        cutoff = len(temp_data)*3/4
        train_data = train_data + temp_data[:cutoff]
        train_targets = train_targets + temp_targets[:cutoff]
        test_data = test_data + temp_data[cutoff:]
        test_targets = test_targets + temp_targets[cutoff:]

        ncat+=1

      print "Finish reading corpus. "

      trainfeats = [(featx(preprocess(train_data[i],'words')), train_targets[i]) for i in range(len(train_data))]
      trainfeats2 = [(preprocess(train_data[i],'dict'), train_targets[i]) for i in range(len(train_data))]
      testfeats = [(featx(preprocess(test_data[i],'words')), test_targets[i]) for i in range(len(test_data))]
      testfeats2 = [(preprocess(test_data[i],'dict'), test_targets[i]) for i in range(len(test_data))]


      #sys.exit()  

    # scikit NB classifier
    print "Scikit classifier: "
    count_vect = CountVectorizer()
    X_train = count_vect.fit_transform(train_data)
    clf = MultinomialNB().fit(X_train, train_targets)     

    X_new = count_vect.transform(test_data)
    predicted = clf.predict(X_new)
    print 'Raw counts accuracy: ',np.mean(predicted == test_targets) 

    tf_transformer = TfidfTransformer(use_idf=False).fit(X_train)
    X_train_tf = tf_transformer.transform(X_train)
    clf = MultinomialNB().fit(X_train_tf, train_targets) 

    X_new = tf_transformer.transform(count_vect.transform(test_data))
    predicted = clf.predict(X_new)
    print 'TF accuracy: ',np.mean(predicted == test_targets) 

    tf_transformer = TfidfTransformer().fit(X_train)
    X_train_tf = tf_transformer.transform(X_train)
    clf = MultinomialNB().fit(X_train_tf, train_targets) 
    print clf.feature_log_prob_

    X_new = tf_transformer.transform(count_vect.transform(test_data))
    predicted = clf.predict(X_new)
    print 'Tfidf accuracy: ',np.mean(predicted == test_targets) 



    # NLTK classifier
    print "NLTK classifier: "
    classifier = NaiveBayesClassifier.train(trainfeats2)
    print 'Raw words accuracy:', nltk.classify.util.accuracy(classifier, testfeats2)

    classifier = NaiveBayesClassifier.train(trainfeats)
    refsets = collections.defaultdict(set)
    testsets = collections.defaultdict(set)
 
    for i, (feats, label) in enumerate(testfeats):
            refsets[label].add(i)
            observed = classifier.classify(feats)
            testsets[observed].add(i)
 
    print 'accuracy:', nltk.classify.util.accuracy(classifier, testfeats)
    #print "--> "+str(classifier)+"\n"
    #print str(testfeats)
    print 'pos precision:', nltk.metrics.precision(refsets['pos'], testsets['pos'])
    print 'pos recall:', nltk.metrics.recall(refsets['pos'], testsets['pos'])
    print 'neg precision:', nltk.metrics.precision(refsets['neg'], testsets['neg'])
    print 'neg recall:', nltk.metrics.recall(refsets['neg'], testsets['neg'])
    classifier.show_most_informative_features()


def word_feats(words):
    return dict([(word, True) for word in words])


def preprocess(text, result_type):

   words=word_tokenize(text)
   #words = re.split(u'(?u)\\b\\w\\w+\\b', text.lower())
   if result_type=="text":
     resp = ""
     for word in words:
       resp+=word+" "
   elif result_type=="dict":
     resp = dict([(word, True) for word in words])
     #print resp
   else:
     resp = words
   return resp


 
#dataset="movies"
dataset="20newsgroups-5"
#dataset="20newsgroups"
#dataset="/home/rdorado/data/20_newsgroups"
#dataset="/home/rdorado/data/ohsumed-all"
#dataset="/home/rdorado/data/Reuters21578-Apte-115Cat"
#dataset="/home/rdorado/data/webkb"

print 'evaluating single word features'
evaluate_classifier(word_feats,dataset,encod='latin-1')



'''
word_fd = FreqDist()
label_word_fd = ConditionalFreqDist()

for word in movie_reviews.words(categories=['pos']):
    #word_fd.inc(word.lower())
    word_fd[word.lower()]+=1
    #label_word_fd['pos'].inc(word.lower())
    label_word_fd['pos'][word.lower()]+=1
 
for word in movie_reviews.words(categories=['neg']):
    #word_fd.inc(word.lower())
    word_fd[word.lower()]+=1 
    #label_word_fd['neg'].inc(word.lower())
    label_word_fd['neg'][word.lower()]+=1

pos_word_count = label_word_fd['pos'].N()
neg_word_count = label_word_fd['neg'].N()
total_word_count = pos_word_count + neg_word_count

word_scores = {}
 
for word, freq in word_fd.iteritems():
    pos_score = BigramAssocMeasures.chi_sq(label_word_fd['pos'][word], (freq, pos_word_count), total_word_count)
    neg_score = BigramAssocMeasures.chi_sq(label_word_fd['neg'][word], (freq, neg_word_count), total_word_count)
    word_scores[word] = pos_score + neg_score
 
best = sorted(word_scores.iteritems(), key=lambda (w,s): s, reverse=True)[:10000]
bestwords = set([w for w, s in best])

def best_word_feats(words):
    return dict([(word, True) for word in words if word in bestwords])

print 'evaluating best word features'
evaluate_classifier(best_word_feats,dataset)



def best_bigram_word_feats(words, score_fn=BigramAssocMeasures.chi_sq, n=200):
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    d = dict([(bigram, True) for bigram in bigrams])
    d.update(best_word_feats(words))
    return d
 
print 'evaluating best words + bigram chi_sq word features'
evaluate_classifier(best_bigram_word_feats,dataset)

'''








