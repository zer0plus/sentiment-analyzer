import pandas as pd
from pandas.io.json import json_normalize
# make a function to Preprocess the text data by removing punctuation, special characters, and stop words, and by performing stemming or lemmatization to 
# reduce each word to its base form. 

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk
import spacy
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# import lancaster
from nltk.stem import LancasterStemmer
#import snowballStemmer
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('wordnet')
nlp = spacy.load('en_core_web_sm')

class sentiment_analyzer:
    # CHANGE REWARD_VALUE TO HELPFULNESS RATING AND STAR RATING AS ACTION AND MAKE ALPHA CONSTANT AND TRY AGAIN
    # TRAINING_SIZE = 200000
    TRAINING_SIZE = 40000
    TOTAL_REVIEW_SIZE = 50000
    def __init__(self, reviewsFilePath):
        self.qTable = {}
        self.trialSet = []
        self.discount_rate_GAMMA = 0.7
        df = pd.read_json(reviewsFilePath, lines=True)

        # Splits all the reviews into 2 sets for training and testing
        reviewText = df["reviewText"].head(self.TOTAL_REVIEW_SIZE).to_numpy()
        star_rating = df["overall"].head(self.TOTAL_REVIEW_SIZE).to_numpy()
        helpfulness_rating = df["helpful"].head(self.TOTAL_REVIEW_SIZE).to_numpy()
        training_reviews = reviewText[0:self.TRAINING_SIZE]
        training_stars = star_rating[0:self.TRAINING_SIZE]
        training_labels = []
        testing_reviews = reviewText[self.TRAINING_SIZE:]
        testing_stars = star_rating[self.TRAINING_SIZE:]
        testing_labels = []

        # /5
        for star in (training_stars):
            if int(star) > 4:
                training_labels.append(1)
            elif int(star) < 2:
                training_labels.append(0)
            else:
                training_labels.append(-1)

        for star in (testing_stars):
            if int(star) > 4:
                testing_labels.append(1)
            elif int(star) < 2:
                testing_labels.append(0)
            else:
                testing_labels.append(-1)

        cleaned_training_reviews = []
        cleaned_training_labels = []
        cleaned_testing_reviews = []
        cleaned_testing_labels = []
        for i, v in enumerate(training_labels):
            if v != -1:
                cleaned_training_reviews.append(training_reviews[i])
                cleaned_training_labels.append(training_labels[i])
        for i, v in enumerate(testing_labels):
            if v != -1:
                cleaned_testing_reviews.append(testing_reviews[i])
                cleaned_testing_labels.append(testing_labels[i])
        cleaned_training_reviews = np.array(cleaned_training_reviews)
        cleaned_testing_reviews = np.array(cleaned_testing_reviews)
        cleaned_testing_labels = np.array(cleaned_testing_labels)
        cleaned_training_labels = np.array(cleaned_training_labels)
        # Below code turns each word into a tokenized number and then stores each review as an array of numbers
        self.tokenizer = Tokenizer(oov_token="<OOV>")
        self.tokenizer.fit_on_texts(cleaned_training_reviews)
        word_index = self.tokenizer.word_index
        training_tokenized_reviews = self.tokenizer.texts_to_sequences(cleaned_training_reviews)
        padded_training_tokenized_reviews = pad_sequences(training_tokenized_reviews, padding='post')
        padded_training_tokenized_reviews = np.array(padded_training_tokenized_reviews)
        # For above code each sentence is padded to be the same length of tokenized words representated as a number
        MAX_VOCAB = padded_training_tokenized_reviews.shape[0]
        self.MAX_SENTENCE_LEN = padded_training_tokenized_reviews.shape[1]
        testing_tokenized_reviews = self.tokenizer.texts_to_sequences(cleaned_testing_reviews)
        padded_testing_tokenized_reviews = pad_sequences(testing_tokenized_reviews, padding='post', maxlen=self.MAX_SENTENCE_LEN)
        padded_testing_tokenized_reviews = np.array(padded_testing_tokenized_reviews)

        self.model = tf.keras.Sequential([
            tf.keras.layers.Embedding(input_dim=100000, output_dim=16, input_length=self.MAX_SENTENCE_LEN),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        num_epochs = 30
        history = self.model.fit(padded_training_tokenized_reviews, cleaned_training_labels, epochs=num_epochs, validation_data=(padded_testing_tokenized_reviews, cleaned_testing_labels), verbose=2)
        
    def analyzeText(self, review_text_arr):
        testSeq = self.tokenizer.texts_to_sequences(review_text_arr)
        paaaaaaaaaad = pad_sequences(testSeq, maxlen=self.MAX_SENTENCE_LEN)
        print(self.model.predict(paaaaaaaaaad))
