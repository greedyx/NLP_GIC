import pickle
import string
from collections import Counter
import numpy as np
import re
from nltk import ngrams
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pandas as pd



# 只计算test的单词数目

def remove_all(target_list):
    # remove all certain values in a list
    delete_element = ['inc', 'corp', 'lp','sa','ag','mr', 'co', 'plc']
    return [i for i in target_list if i not in delete_element]


def sub_words(sentence):
    # set which words should be transfered into a certain word
    # use | to adept case with many similar words
    # 记得无论原词还是新词，后面加一个空格
    replacements = [
        (r'([1-2][0-9]{3}|[1][9][0-9]{2}) ','y '),  # 年份，一定得放最前面
        (r'(\d+ millions)', 'mn '),  # millions
        (r'(\d+ billions)', 'bn '),  # billions
        (r'[0-9]{4}-[0-9]{2}','y0m '), # xxxx-xx 年-月
        (r"[-+]?\d*\.\d+|\d+ ", 'f '),  # 小数
        (r'\d+ ', 'n '),  # 整数
        ('%', 'pct '),
        ('don\'t ', 'do not '),
        ('\'s ', ' '), # apply's shares -> apple shares

        ('chief executive ', 'ceo '),
        ('jpmorgan chase |jpmorgan ', '0company0 '),
        ('bank of america |bank of america merrill lynch |bank of america corp |bank of america corporation |bofa ',
         '0company0 '),
        ('goldman sachs |goldman sachs group |goldman ', '0company0 '),
        ('apple ','aapl '),
        ('amazon.com |amazon-com |amazon ','0company0 '),
        ('amazon |apple |facebook |google |ibm |intel |lyft |microsoft |netflix '
         '|qualcomm |snap |tesla |uber ','0company0 '),
        ('bank of america |barclays |citigroup |credit suisse |deutsche Bank |goldman sachs |hsbc '
         '|jpmorgan chase |morgan stanley |ubs |wells fargo ','0company0 ')
    ]
    for old, new in replacements:
        sentence = re.sub(old, new, sentence)
    return sentence


class FreqToken:
    def __init__(self, cleaned_news, top=5000):
        self.cleaned_news = cleaned_news
        self.translator = str.maketrans('', '', string.punctuation)  # To get rid of the punctuations
        self.frequent_grams = {}
        self.top = top
        self.stopWords = set(stopwords.words('english'))


    def __call__(self):
        print("processing freq_token fpr evaluation set")
        print('\n')
        for date in self.cleaned_news.keys():
            # if date == np.datetime64('2018-07-08'):
            sen_tokens = []
            for content in self.cleaned_news[date]:
                for sentence in sent_tokenize(content):
                    # Here should add some functions to do data clean
                    sentence = sub_words(sentence)
                    temp = word_tokenize(sentence.translate(self.translator))
                    temp = remove_all(temp)
                    temp = self.stop_words_and_stem(temp)
                    sen_tokens += [temp]
            flat_tokens = []
            for tokens in sen_tokens:
                flat_tokens += list(ngrams(tokens, 4))
            today_frequent_tokens = self.take_frequent_tokens(flat_tokens)
            self.frequent_grams[np.datetime64(date)] = today_frequent_tokens
        return

    def take_frequent_tokens(self, flat_tokens):
        today_frequent_tokens = {}
        counter = Counter(flat_tokens)
        top_tokens = counter.most_common(self.top)
        for pair in top_tokens:
            if pair[1] >= 1:
                today_frequent_tokens[pair[0]] = pair[1]
        return today_frequent_tokens
        # self.frequent_tokens = list(list(zip(*top_tokens))[0])

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


def run(profile=""):
    frequency = 'M'
    candidates_tech = ['Amazon', 'Apple', 'Facebook', 'Google', 'IBM', 'Intel', 'Lyft', 'Microsoft', 'Netflix',
                       'Qualcomm', 'Snap', 'Tesla', 'Uber']
    candidates_bank = ['Bank of America', 'Barclays', 'Citigroup', 'Credit Suisse', 'Deutsche Bank', 'Goldman Sachs',
                       'HSBC',
                       'JPMorgan Chase', 'Morgan Stanley', 'UBS', 'Wells Fargo']
    input_dir = "D:/GIC_Project/Tingjun/data/cleaned/"
    output_dir = "D:/GIC_Project/Tingjun/data/3grams/bank/"

    for company in candidates_bank:
        pickle_in = open(input_dir + "{}_{}.pickle".format(company,frequency), "rb")
        cleaned_news = pickle.load(pickle_in)
        pickle_in.close()

        freToken = FreqToken(cleaned_news)
        freToken()

        pickle_out = open(output_dir + "{}_{}_3grams.pickle".format(company,frequency), "wb")
        pickle.dump(freToken.frequent_grams, pickle_out)
        pickle_out.close()
        print("Finished dumping " + company + "/3_freq_grams" + profile + ".pickle\n")
        print('\n')

if __name__ == '__main__':
    run()
