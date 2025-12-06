import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
import pickle

mental_health_data = pd.read_csv('data/Combined Data.csv')
subject_data = pd.read_csv('data/Subject Data.csv')

# removes rows with missing text or label
mental_health_data = mental_health_data.dropna(subset=['statement', 'status'])
subject_data = subject_data.dropna(subset=['text', 'subject'])

# import nltk
# nltk.download('stopwords')
# print(subject_data.head(3))
# print(mental_health_data.columns)

stemmer = SnowballStemmer('english') # converts words to their route
words = stopwords.words('english')   # cuts not useful words

X_train_health, X_test_health, y_train_health, y_test_health = train_test_split(mental_health_data['statement'], mental_health_data['status'], test_size=0.2, random_state=42)
X_train_subject, X_test_subject, y_train_subject, y_test_subject = train_test_split(subject_data['text'], subject_data['subject'], test_size=0.2, random_state=42)

pipeline_health = Pipeline([('vect', TfidfVectorizer(ngram_range=(1, 2), stop_words='english', # "a bag of words"
                                              max_features=50000, sublinear_tf=True)),  # certain weight for every word
                     ('chi', SelectKBest(chi2, k=5000)), # feature selection
                     ('clf', LinearSVC(C=1.0, penalty='l2', dual=True, max_iter=2000))]) # classification

pipeline_subject = Pipeline([('vect', TfidfVectorizer(ngram_range=(1, 2), stop_words='english', # "a bag of words"
                                              max_features=50000, sublinear_tf=True)),  # certain weight for every word
                     ('clf', LinearSVC(C=1.0, penalty='l2', dual=True, max_iter=2000))]) # classification

model_health = pipeline_health.fit(X_train_health, y_train_health)
model_subject = pipeline_subject.fit(X_train_subject, y_train_subject)

# mental health
vect_health = pipeline_health.named_steps['vect']
chi_health = pipeline_health.named_steps['chi']
clf_health = pipeline_health.named_steps['clf']

feature_names_health = vect_health.get_feature_names_out()
feature_vector_health = [feature_names_health[i] for i in chi_health.get_support(indices=True)]
target_names_health = clf_health.classes_

# subject model
vect_subject = pipeline_subject.named_steps['vect']
clf_subject = pipeline_subject.named_steps['clf']

feature_names_subject = vect_subject.get_feature_names_out()
target_names_subject = clf_subject.classes_


# print('top 10 keywords per class')
# for i, label in enumerate(target_names):
#     top10 = np.argsort(np.abs(clf.coef_[i]))[-10:]
#     print(feature_names[top10])

print('health model accuracy' + str(model_health.score(X_test_health, y_test_health)))
print(model_health.predict(['i`m scared i will fail exam.']))

print('subject model accuracy' + str(model_subject.score(X_test_subject, y_test_subject)))
print(model_subject.predict(['i`m scared i will fail math exam.']))

# pickle file of the models
# pickle.dump(pipeline_health, open('health.pkl', 'wb'))
# pickle.dump(pipeline_subject, open('subject.pkl', 'wb'))