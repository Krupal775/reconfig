from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd
import re
import string
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn import svm
from sklearn.preprocessing import MaxAbsScaler
from sklearn.feature_extraction.text import TfidfVectorizer

# initialzing flask app
app = Flask(__name__)

filename0 = 'svm_model.sav'
filename1 = 'tfidf.pickle'
filename2 = 'svm_final.sav'
filename3 = 'cat_subcat_data.pickle'
category_model = pickle.load(open(filename0, 'rb'))
tfidf = pickle.load(open(filename1, 'rb'))
subcategory_model = pickle.load(open(filename2, 'rb'))
cat_subcat_dict = pickle.load(open(filename3, 'rb'))

stop_words = set(stopwords.words('english'))

#adding some unwanted words that are not already in the stopwords set
stop_words.add('us')
stop_words.add('thank')
stop_words.add('thankyou')

max_len = 25
words = set(nltk.corpus.words.words())
lemmatizer = WordNetLemmatizer()
vectorizer = TfidfVectorizer()
scaler = MaxAbsScaler()

#functions for text cleaning
def remove_URL(text):
    url = re.compile(r"https?://\S+|www\.\S+")
    return url.sub(r"", text)

def remove_num(text):
    num = re.compile(r"[^\w\s]z")
    return num.sub(r"", text)
        
def remove_punct(text):
    translator = str.maketrans("", "", string.punctuation)
    return text.translate(translator)

def remove_stopwords(text):
    filtered_text = ""
    for sentence in sent_tokenize(text):
        for word in word_tokenize(sentence):
            word = word.lower()
            if word not in stop_words and (word in words or word.isalpha()) and len(word) < max_len:
                filtered_text += word + " "
    return filtered_text

def lemmatize_text(text):
    lemmatized_text = ""
    for sentence in sent_tokenize(text):
        for word in word_tokenize(sentence):
            lemmatized_text += lemmatizer.lemmatize(word) + " "
    return lemmatized_text

#home function
@app.route('/')

def home():
    return render_template('mail.html')

#function for predicting category and subcategory
def predict_topic(text):

    text = lemmatize_text(remove_stopwords(remove_punct(remove_num(remove_URL(text)))))
    X = [text]

    #vectorizing and scaling
    X = tfidf.transform(X).toarray()
    transformer = scaler.fit(X)
    X = transformer.transform(X)

    category_pred = category_model.predict(X)[0]
    
    if category_pred == 'Other':
        subcategory_pred = 'Other'
        return category_pred, subcategory_pred
        

    #finding decision function of classes
    p = np.array(subcategory_model.decision_function(X))
    subcategory_prob = np.exp(p)/np.sum(np.exp(p), axis = 1, keepdims = True) 

    #converting into dataframe
    subcategory_prob_df = pd.DataFrame(subcategory_prob, columns = sorted(subcategory_model.classes_))
    subcategories = cat_subcat_dict[category_pred]
    max_decision_score = -1

    #selecting subcategory with the best decision score
    for subcategory in subcategories:
        if subcategory in subcategory_model.classes_ and subcategory_prob_df.loc[0, subcategory] > max_decision_score:
            max_decision_score = subcategory_prob_df.loc[0, subcategory]
            subcategory_pred = subcategory

    return category_pred, subcategory_pred

@app.route('/predict', methods=['POST'])

def predict():
    text = request.form['email'] 
    category_pred, subcategory_pred = predict_topic(text)

    #update the text, predicted category and subcategory in the new dataset
    df_new = pd.read_csv('New_Data.csv')
    df_new_record = pd.DataFrame.from_records([{'Category': category_pred, 'Subcategory': subcategory_pred, 'Text': text}])
    df_updated = pd.concat([df_new, df_new_record], ignore_index = True)
    df_updated.to_csv('New_Data.csv', index = False)

    return render_template('mail.html', category_pred = category_pred, subcategory_pred = subcategory_pred)


@app.route('/reconfigure', methods=['POST'])

def re_train():

    #combining the new dataset with the original train dataset
    df_old = pd.read_csv('Email_Clean.csv')
    df_new = pd.read_csv('New_Data.csv')
    df_train_new = pd.concat([df_old, df_new], ignore_index = True)

    #empty the new dataset for storing further records of upcoming emails
    df_new = df_new[0:0]

    df_train_new.to_csv('Email_Clean.csv', index = False)
    df_new.to_csv('New_Data.csv', index = False)

    #re-training models with updated dataset
    df = pd.read_csv('Email_Clean.csv')
    df['Text'] = df['Text'].apply(lambda val : remove_stopwords(remove_punct(remove_num(remove_URL(val)))))
    df['Text'] = df['Text'].apply(lambda val : lemmatize_text(val))

    X = df.Text
    y = df.Category

    #vectorization and scaling
    X_vec = vectorizer.fit_transform(X).toarray()

    transformer = scaler.fit(X_vec)
    X_scaled = transformer.transform(X_vec)

    #training model again for category prediction
    model = svm.LinearSVC(C = 0.1)
    model.fit(X_scaled, y)

    #saving files as pickles
    file0 = 'svm_model.sav'
    pickle.dump(model, open(file0, 'wb'))

    file1 = 'tfidf.pickle'
    pickle.dump(vectorizer, open(file1, 'wb'))

    #updating dictionary
    cat_subcat_dict = {}
    categories = np.array(df['Category'].unique())

    for category in categories:
        new_df = df[df['Category'] == category]
        subcategories = []
        for index, row in new_df.iterrows():
            if row['Subcategory'] not in subcategories:
                subcategories.append(row['Subcategory'])
        cat_subcat_dict[category] = subcategories
    
    cat_subcat_dict['Other'] = ['Other']

    y = df.Subcategory

    #training model again for subcategory prediction
    subcategory_model = svm.SVC(C = 10, gamma = 0.01, kernel = 'rbf')
    subcategory_model.fit(X_scaled, y)

    #saving files as pickles
    file3 = 'svm_final.sav'
    file4 = 'cat_subcat_data.pickle'
    pickle.dump(subcategory_model, open(file3, 'wb'))
    pickle.dump(cat_subcat_dict, open(file4, 'wb'))

    return "SUCCESS"
    
if __name__ == "__main__":
    app.run(debug=True)