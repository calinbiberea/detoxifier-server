import pickle as pkl
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
import numpy as np

with open('RF.pkl', 'rb') as f:
    pipe = pkl.load(f)


def fake_predict(comment):
    res = pipe.predict([comment])
    return res[0]
