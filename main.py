# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
df = pd.read_csv(r"C:\Users\HP\Desktop\New ML\Bag-of-words\spam.csv")
print(df.head())
print(df.Category.value_counts())
df['spam'] = df['Category'].apply(lambda x:1  if x == 'spam' else 0)
print(df.head())
# Train test split
X_train,X_test,y_train,y_test = train_test_split(df.Message,df.spam,test_size=0.2,random_state=42)
print(X_train.shape)
print(X_test.shape)
print(X_train[:4])
print(y_train[:4])
# Create bag of words representation using CountVectorizer
v = CountVectorizer()
X_train_cv = v.fit_transform(X_train)
X_test_cv = v.transform(X_test)
X_train_cv.toarray()[:2][0]
print(X_train.shape)
print(v.get_feature_names_out()[1000])
print(v.vocabulary_)
X_train_np = X_train_cv.toarray()
print(X_train_np)
print(np.where(X_train_np[0]!=0))
# Naive bayes classifier
model = MultinomialNB()
model.fit(X_train_cv,y_train)
y_pred = model.predict(X_test_cv)
print(classification_report(y_test,y_pred))
# Test on random datapoint
message = {"Click here to win 50000$"}
message_cnt = v.transform(message)
print(model.predict(message_cnt))
