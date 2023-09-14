import nltk
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

nltk.download('stopwords')
stopwords.words('english')


def message_cleaning(message):
    test_punc_removed = [char for char in message if char not in string.punctuation]
    test_punc_removed_join = ''.join(test_punc_removed)
    test_punc_removed_join_clean = [word for word in test_punc_removed_join.split() if
                                    word.lower() not in stopwords.words('english')]
    return test_punc_removed_join_clean


# When you run it for the first time, it will take a bit to load.
# This is because we're loading the emails.csv, treating the data, and training the model. Then you can send thr cURL.

# Prep the data to train
spam_df = pd.read_csv("emails.csv")
spam_df['length'] = spam_df['text'].apply(len)
ham = spam_df[spam_df['spam'] == 0]
spam = spam_df[spam_df['spam'] == 1]
spam_df_clean = spam_df['text'].apply(message_cleaning)
vectorizer = CountVectorizer(analyzer=message_cleaning)
spamham_countvectorizer = vectorizer.fit_transform(spam_df['text'])
NB_classifier = MultinomialNB()
label = spam_df['spam'].values
NB_classifier.fit(spamham_countvectorizer, label)

# Testing
testing_sample = ['Free money!!!', "Hi Kim, Please let me know if you need any further information. Thanks"]
testing_sample_countvectorizer = vectorizer.transform(testing_sample)
test_predict = NB_classifier.predict(testing_sample_countvectorizer)
print(test_predict)

# One More Test
testing_sample = ['Hello, I am Ryan, I would like to book a hotel in Bali by January 24th', 'money viagara!!!!!']
testing_sample_countvectorizer = vectorizer.transform(testing_sample)
test_predict = NB_classifier.predict(testing_sample_countvectorizer)
print(test_predict)

# Start API
app = FastAPI()


# Model definition
class Email(BaseModel):
    SubjectEmail: str


@app.post('/')
async def email_spam_prediction(item: Email):
    message_clean = message_cleaning(item.SubjectEmail)
    message_vectorizer = vectorizer.transform(message_clean)
    text_prediction = NB_classifier.predict(message_vectorizer)
    print(text_prediction)
    return {"Is this subject a Spam?": str(text_prediction[0])}
