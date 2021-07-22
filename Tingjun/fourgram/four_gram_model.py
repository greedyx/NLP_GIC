"""
This is the module to train the ngram model, and generate the fitted model for each day.
"""
import numpy as np
import os
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE
import string
import pickle
import time
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
# from nltk import Text
# from data_cleaner import DataCleaner
# from nltk.util import pad_sequence
# from nltk.util import ngrams
# from nltk import FreqDist


class FourGramModel:
    def __init__(self, input_data_frame, input_date, train_window):
        """
        To instantiate the 4-gram model, we need to provide the dataframe , and the date of interest

        :param input_data_frame: input data_frame, which has the same format as the output oof DataCleaner
        :type input_data_frame: data_frame
        :param input_date: date of interesting
        :type input_date: np.datetime64,['D']
        :patram train_window: int
        # Will use how many days data to train model before interesting date
        """
        self.input_data_frame = input_data_frame
        self.input_date = input_date
        self.training_window = train_window
        self.train = []
        self.vocab = []
        self.lm = MLE(4)
        self.translator = str.maketrans('', '', string.punctuation)  # To get rid of the punctuations
        self.stopWords = set(stopwords.words('english'))
        return

    def __call__(self):
        start = time.time()
        print("Fitting model for date: " + str(self.input_date) + "\n")
        self.train_model()
        end = time.time()
        print("Model for date: " + str(self.input_date) + " has been fitted \n" +
              "Time taking: " + str(end-start) + "\n")

    def stop_words_and_stem(self, word_list):
        """
        This is the function which will remove the stop words and do the word stemming
        :param word_list: a list of words to be processed
        :return:
        """
        return_list = []
        ps = PorterStemmer()
        for w in word_list:
            if w not in self.stopWords:
                return_list.append(ps.stem(w))
        return return_list

    def train_model(self):
        """
        Train the model with data one week before

        :return: A fitted 4gram model
        """
        training_text = []
        training_series = self.input_data_frame[(self.input_data_frame.index < self.input_date) &\
                    (self.input_data_frame.index > pd.Timestamp(self.input_date) - pd.DateOffset(months = self.training_window))]
#self.input_date - self.training_window)]
        for idx in training_series.index:
            for content in training_series[idx]:
                training_text += [self.stop_words_and_stem(word_tokenize(sentence.translate(self.translator)))
                                  for sentence in sent_tokenize(content)]
        every_gram, vocab = padded_everygram_pipeline(4, training_text)  # This will generate unigram, bigram,
        four_gram = []
        for sent in list(every_gram):
            for item in sent:
                if len(item) == 4:
                    four_gram.append(item)
        train = [four_gram]
        self.lm.fit(train, vocab)


#  Only executed when this file is called directly, this part of code is for debug purpose only
if __name__ == '__main__':
    train_window = 24
    frequency = 'M'

    candidates = ['Amazon', 'Apple']
    temp_series = []
    for company in candidates:
        dict_pickle_path = "../data/cleaned/{}_{}.pickle".format(company,frequency)
        pickle_in = open(dict_pickle_path, "rb")
        temp_series.append( pickle.load(pickle_in))
        pickle_in.close()
    df = pd.concat(temp_series,axis = 1)
    df = df.apply(lambda s: s.fillna({i: [] for i in df.index}))
    processed_news_dataframe = df.apply(lambda x: x.sum(), axis=1)

    model_1 = FourGramModel(processed_news_dataframe, np.datetime64('2020-03-04'),train_window)
    model_1()

    result_dir = "E:\\GIC_Project\\Tingjun\\data\\2_models\\"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    pickle_out = open(result_dir + "\\{}{}_model.pickle".format(frequency,train_window), "wb")
    pickle.dump({np.datetime64('2020-03-04 00:00:00'): model_1.lm}, pickle_out)
    pickle_out.close()
