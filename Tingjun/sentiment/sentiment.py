import Load_MasterDictionary as LM

MASTER_DICTIONARY_FILE = r'../data/sentiment_dic/LoughranMcDonald_MasterDictionary_2018.csv'


class Sentiment:
    def __init__(self):
        self.lm_dictionary = LM.load_masterdictionary(MASTER_DICTIONARY_FILE, True)

    def pos_neg(self, ngram):
        pos, neg = 0, 0
        for token in ngram:
            token = token.upper()
            if not token.isdigit() and len(token) > 1 and token in self.lm_dictionary:
                pos += self.lm_dictionary[token].positive
                neg += self.lm_dictionary[token].negative
        return pos > 0 and neg == 0, neg > 0 and pos == 0


if __name__ == '__main__':
    import os
    import sys
    rootpath = str(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(rootpath)
    test = Sentiment()
    print(test.pos_neg(['ABANDON']))  # (False, True)
    print(test.pos_neg(['ABLE']))   #(True, False)
    print(test.pos_neg(['ABANDON','ABLE']))  #(False, False)
