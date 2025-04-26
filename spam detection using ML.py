#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd # pandas library is used for data manipulation


# In[2]:


import re  #re = regular expression fior calculations and pattern making


# In[3]:


import nltk #nltk = natural lanhuage toolkit, is a part of nlp


# In[4]:


from nltk.corpus import stopwords  #corpus means a collection of written or spoken texts. i,am,was,were,will


# In[5]:


from nltk.stem import PorterStemmer  #running = run


# In[6]:


nltk.download("stopwords")  #downloading stopwords


# In[7]:


stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))


# In[9]:


df = pd.read_csv("D:\\ML projects work\\01. Project Spam Detection - DATASET.csv",encoding = 'latin-1')[["v1","v2"]]


# In[10]:


df.head()  #first five rows of the given dataset


# In[11]:


df.tail(10)


# In[12]:


df.columns = ['label','message']


# In[13]:


df.head()


# In[14]:


df.tail()


# In[15]:


df['label']= df["label"].map({"ham":0, "spam":1})   #one hot encoding


# In[16]:


df.head()


# In[38]:


def preprocess_text(text):
    text = re.sub(r"\W", " ", text) #remove special symbols
    text = text.lower() #converting all the text into lowercase
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    #remove stopwords and stem words
    return " ".join(words)


# In[39]:


df["cleaned_message"] = df["message"].apply(preprocess_text)


# In[40]:


df.head()


# In[41]:


# Importing Data Science - ML Libraries using SKLEARN


# In[42]:


from sklearn.feature_extraction.text import TfidfVectorizer # converting text to numerical format


# In[43]:


from sklearn.model_selection import train_test_split # distributing data into train and test for predict


# In[44]:


from sklearn.linear_model import LogisticRegression


# In[45]:


from sklearn.metrics import accuracy_score, classification_report # check the accuracy of ML model


# In[46]:


df.head()


# In[47]:


vectorizer = TfidfVectorizer(max_features = 3000)
X = vectorizer.fit_transform(df["cleaned_message"]) # input data


# In[48]:


y = df["label"] # output data 


# In[49]:


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, random_state = 42)


# In[50]:


model = LogisticRegression()


# In[51]:


model.fit(X_train, y_train) # using train data we can predict the test data 


# In[52]:


y_pred = model.predict(X_test)


# In[53]:


print(f"accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")


# In[54]:


print(classification_report(y_test, y_pred))


# In[58]:


def predict_email(email_text):
    processed_text = preprocess_text(email_text)
    processed_data = preprocess_text(email_text)
    vectorized_text = vectorizer.transform([processed_text])
    prediction = model.predict(vectorized_text)
    return "Spam" if prediction[0]==1 else "Ham - Not Spam"


# In[59]:


email = """
Hello Chaitanya,
Java is moving at a pace we've never seen before. With faster release cycles and shorter LTS versions, it’s no longer
just a “legacy” platform — it’s one of the most forward-moving languages in the industry. That means more tools at
your disposal: from virtual threads that simplify scalability, to records and pattern matching that make code cleaner
    and safer.The challenge? Knowing what’s actually useful to you, your team, and your current codebase. Whether 
    it’s understanding how to use multi-release JARs, making sense of Project Loom, or seeing how Project Panama will 
    change native interop — this is your opportunity to get clear answers from the people shaping the future of Java."""


# In[60]:


print(f"Email: {email}/n Prediction: {predict_email(email)}")


# In[ ]:




