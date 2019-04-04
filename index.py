from flask import *
import nltk
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
import pickle
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/filtered",methods=["GET"])
def filtered():
    q = request.args.get('input1')
    print(q) # read data from field
    nltk.download("stopwords") # stopword required
    p = joblib.load('model123.pkl')
    p1 = joblib.load('Countvectorizer.pkl')
    p2= joblib.load('feature.pkl')
    ps=PorterStemmer()
    # check the sentence
    cr1=[]
    #cv=CountVectorizer(vocabulary=p1)
    Emotion1=re.sub('[^a-zA-Z]'," ",q)
    Emotion1=Emotion1.lower()
    Emotion1=Emotion1.split()
    Emotion1=[ps.stem(word) for word in Emotion1 if not word in set(stopwords.words("english"))]
    cr1.append(" ".join(Emotion1))
    x_p1=p1.transform(cr1).toarray()
    x_p2= p2.transform(x_p1)
    y_predict=p.predict(x_p2)
    print(y_predict)
    if(y_predict==1):
    	return "Happy"
    elif(y_predict==2):
    	return "Neutral"
    elif(y_predict==3):
    	return "Sad"
    elif(y_predict==4):
    	return "worry"
    else:
    	return "Anger" 
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)