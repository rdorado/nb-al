from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2
from sklearn.preprocessing import LabelBinarizer
import numpy as np

filenames = ['Austen_Emma.txt','Austen_Pride.txt','Austen_Sense.txt','CBronte_Jane.txt','CBronte_Professor.txt','CBronte_Villette.txt']
austen_indices, cbronte_indices = [0,1,2], [3,4,5]

raw_texts = []
for fn in filenames:
     with open("/home/rdorado/data/dariah/data/austen-bronte/"+fn) as f:
         text = f.read()
         text = text.replace('_', '')  # remove underscores (italics)
         raw_texts.append(text)


vectorizer = CountVectorizer(input='content')
dtm = vectorizer.fit_transform(raw_texts)
vocab = np.array(vectorizer.get_feature_names())
dtm = dtm.toarray()


rates = 1000.0 * dtm / np.sum(dtm, axis=1, keepdims=True)
austen_rates = rates[austen_indices, :]
cbronte_rates = rates[cbronte_indices, :]

austen_rates_avg = np.mean(austen_rates, axis=0)
cbronte_rates_avg = np.mean(cbronte_rates, axis=0)

distinctive_indices = (austen_rates_avg * cbronte_rates_avg) == 0
ranking = np.argsort(austen_rates_avg[distinctive_indices] + cbronte_rates_avg[distinctive_indices])[::-1]

#print vocab[distinctive_indices][ranking]

#remove words from corpus
dtm = dtm[:, np.invert(distinctive_indices)]
rates = rates[:, np.invert(distinctive_indices)]
vocab = vocab[np.invert(distinctive_indices)]
austen_rates = rates[austen_indices, :]
cbronte_rates = rates[cbronte_indices, :]

austen_rates_avg = np.mean(austen_rates, axis=0)
cbronte_rates_avg = np.mean(cbronte_rates, axis=0)


#analysis of 'keyness'
keyness = np.abs(austen_rates_avg - cbronte_rates_avg)
ranking = np.argsort(keyness)[::-1]
#print vocab[ranking][0:10]


rates_avg = np.mean(rates, axis=0)
keyness = np.abs(austen_rates_avg - cbronte_rates_avg) / rates_avg
ranking = np.argsort(keyness)[::-1]
#print vocab[ranking][0:10]


# **************************************
#    Bayesian ... (LDA?)
# **************************************

#analysis
print np.mean(rates < 4)
print np.mean(rates > 1)
print mquantiles(rates, prob=[0.01, 0.5, 0.99])

def sample_posterior(y1, y2, mu0, sigma20, nu0, delta0, gamma20, tau20, S):
  n1, n2 = len(y1), len(y2)
  mu = (np.mean(y1) + np.mean(y2))/2
  delta = (np.mean(y1) - np.mean(y2))/2
  vars = ['mu', 'delta', 'sigma2']
  chains = {key: np.empty(S) for key in vars}
  for s in range(S):
    a = (nu0+n1+n2)/2
    b = (nu0*sigma20 + np.sum((y1-mu-delta)**2) + np.sum((y2-mu+delta)**2))/2
    sigma2 = 1 / np.random.gamma(a, 1/b)
    mu_var = 1/(1/gamma20 + (n1+n2)/sigma2)
    mu_mean = mu_var * (mu0/gamma20 + np.sum(y1-delta)/sigma2 + np.sum(y2+delta)/sigma2)
    mu = np.random.normal(mu_mean, np.sqrt(mu_var))
    delta_var = 1/(1/tau20 + (n1+n2)/sigma2)
    delta_mean = delta_var * (delta0/tau20 + np.sum(y1-mu)/sigma2 - np.sum(y2-mu)/sigma2)
    delta = np.random.normal(delta_mean, np.sqrt(delta_var))
    chains['mu'][s] = mu
    chains['delta'][s] = delta
    chains['sigma2'][s] = sigma2
  return chains



mu0 = 3
tau20 = 1.5**2
nu0 = 1
sigma20 = 1
delta0 = 0
gamma20 = 1.5**2 
S = 2000 ## number of samples

word = "green"
y1, y2 = austen_rates[:, vocab == word], cbronte_rates[:, vocab == word]
chains = sample_posterior(y1, y2, mu0, sigma20, nu0, delta0, gamma20, tau20, S)
delta_green = chains['delta']

word = "dark"
y1, y2 = austen_rates[:, vocab == word], cbronte_rates[:, vocab == word]
chains = sample_posterior(y1, y2, mu0, sigma20, nu0, delta0, gamma20, tau20, S)
delta_dark = chains['delta']


def delta_confidence(rates_one_word):
  austen_rates = rates_one_word[0:3]
  bronte_rates = rates_one_word[3:6]
  chains = sample_posterior(austen_rates, bronte_rates, mu0, sigma20, nu0, delta0, gamma20, tau20, S)
  delta = chains['delta']
  return np.max([np.mean(delta < 0), np.mean(delta > 0)])

keyness = np.apply_along_axis(delta_confidence, axis=0, arr=rates)
ranking = np.argsort(keyness)[::-1]
# print vocab[ranking][0:10]


# **************************************
# Log likelihood ratio and chi^2 feature selection
# **************************************
green_austen = np.sum(dtm[austen_indices, vocab == "green"])
nongreen_austen = np.sum(dtm[austen_indices, :]) - green_austen
green_cbronte = np.sum(dtm[cbronte_indices, vocab == "green"])
nongreen_cbronte = np.sum(dtm[cbronte_indices, :]) - green_cbronte

green_table = np.array([[green_austen, nongreen_austen], [green_cbronte, nongreen_cbronte]])

# Log likelihood
prob_green = np.sum(dtm[:, vocab == "green"]) / np.sum(dtm)
prob_notgreen = 1 - prob_green
labels = []

for fn in filenames:
  label = "Austen" if "Austen" in fn else "CBronte"
  labels.append(label)

n_austen = np.sum(dtm[labels == "Austen", :])
n_cbronte = np.sum(dtm[labels != "Austen", :])
expected_table = np.array([[prob_green * n_austen, prob_notgreen * n_austen], [prob_green * n_cbronte, prob_notgreen * nongreen_cbronte]])

print expected_table


# same result, but more concise and more general
X = dtm[:, vocab == "green"]
X = np.append(X, np.sum(dtm[:, vocab != "green"], axis=1, keepdims=True), axis=1)
y = LabelBinarizer().fit_transform(labels)
y = np.append(1 - y, y, axis=1)
green_table = np.dot(y.T, X)
print green_table

feature_count = np.sum(X, axis=0, keepdims=True)
class_prob = np.mean(y, axis=0, keepdims=True)
expected_table = np.dot(class_prob.T, feature_count)

G = np.sum(green_table * np.log(green_table / expected_table))


# Pearson’s chi^2 test statistic approximates the log likelihood ratio test 
labels = []
keyness, pvals = chi2(dtm, labels)
ranking = np.argsort(keyness)[::-1]
vocab[ranking][0:10]


# Mutual information

def computeMI(x, y):
    sum_mi = 0.0
    x_value_list = np.unique(x)
    y_value_list = np.unique(y)
    Px = np.array([ len(x[x==xval])/float(len(x)) for xval in x_value_list ]) #P(x)
    Py = np.array([ len(y[y==yval])/float(len(y)) for yval in y_value_list ]) #P(y)
    for i in xrange(len(x_value_list)):
        if Px[i] ==0.:
            continue
        sy = y[x == x_value_list[i]]
        if len(sy)== 0:
            continue
        pxy = np.array([len(sy[sy==yval])/float(len(y))  for yval in y_value_list]) #p(x,y)
        t = pxy[Py>0.]/Py[Py>0.] /Px[i] # log(P(x,y)/( P(x)*P(y))
        sum_mi += sum(pxy[t>0]*np.log2( t[t>0]) ) # sum ( P(x,y)* log(P(x,y)/( P(x)*P(y)) )
    return sum_mi



# Calculates the entropy of the given data set for the target attribute.
def entropy(data, target_attr):

    val_freq = {}
    data_entropy = 0.0

    # Calculate the frequency of each of the values in the target attr
    for record in data:
        if (val_freq.has_key(record[target_attr])):
            val_freq[record[target_attr]] += 1.0
        else:
            val_freq[record[target_attr]]  = 1.0

    # Calculate the entropy of the data for the target attribute
    for freq in val_freq.values():
        data_entropy += (-freq/len(data)) * math.log(freq/len(data), 2) 

    return data_entropy


# Calculates the information gain (reduction in entropy) that would result by splitting the data on the chosen attribute (attr).
def gain(data, attr, target_attr):

    val_freq = {}
    subset_entropy = 0.0

    # Calculate the frequency of each of the values in the target attribute
    for record in data:
        if (val_freq.has_key(record[attr])):
            val_freq[record[attr]] += 1.0
        else:
            val_freq[record[attr]]  = 1.0

    # Calculate the sum of the entropy for each subset of records weighted by their probability of occuring in the training set.
    for val in val_freq.keys():
        val_prob = val_freq[val] / sum(val_freq.values())
        data_subset = [record for record in data if record[attr] == val]
        subset_entropy += val_prob * entropy(data_subset, target_attr)

    # Subtract the entropy of the chosen attribute from the entropy of the whole data set with respect to the target attribute (and return it)
    return (entropy(data, target_attr) - subset_entropy)


