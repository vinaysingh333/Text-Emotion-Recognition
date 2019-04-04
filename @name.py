import pandas as pd 
import numpy as np 
from sklearn.preprocessing import LabelEncoder
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib
dataset=pd.read_csv("main11.csv",encoding='latin1')
X1=dataset.iloc[:,0].values
Y1=dataset.iloc[:,1].values
labelEncoder_X=LabelEncoder()
X1=labelEncoder_X.fit_transform(X1)
cr=[]
for i in range(0,27576):
    lis=Y1[i].split(" ")
    for k in lis:
        try:
            if(k[0]=="@"):
                lis.remove(k)
        except:
            l=1
    Y1[i]=" ".join(lis)
    Emotion=re.sub('[^a-zA-Z]'," ",Y1[i])
    Emotion=Emotion.lower()
    Emotion=Emotion.split()
    ps=PorterStemmer()
    Emotion=[ps.stem(word) for word in Emotion if not word in set(stopwords.words("english"))]
    cr.append(" ".join(Emotion))

cv=CountVectorizer()
x=cv.fit_transform(cr).toarray()
sc = StandardScaler()
xP = sc.fit_transform(x)
classifier = RandomForestClassifier(n_estimators = 90, criterion = 'entropy', random_state = 0)
classifier.fit(xP,X1)
x_p="i am happy today "
cr1=[]
Emotion1=re.sub('[^a-zA-Z]'," ",x_p)
Emotion1=Emotion1.lower()
Emotion1=Emotion1.split()
ps=PorterStemmer()
Emotion1=[ps.stem(word) for word in Emotion1 if not word in set(stopwords.words("english"))]
cr1.append(" ".join(Emotion1))
x_p1=cv.transform(cr1).toarray()
x_p1= sc.transform(x_p1)
y_predict=classifier.predict(x_p1)
print(y_predict)
if(y_predict==1):
    print("Happy")
elif(y_predict==2):
    print("Neutral")
elif(y_predict==3):
    print("Sad")
elif(y_predict==4):
    print("worry")
else:
    print("Anger")
joblib.dump(classifier,"modelllll.pkl")
joblib.dump(cv,"countvectorizer.pkl")
joblib.dump(sc,"feature.pkl")

