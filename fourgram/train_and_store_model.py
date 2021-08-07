from fourgram.four_gram_model import FourGramModel
# from data_cleaner import DataCleaner
import pickle
import numpy as np
import pandas as pd
import os
import time
# from multiprocessing import Pool
# from multiprocessing import Manager
import multiprocessing as mp
from functools import partial

model_dir = '../data/2_models/bank'

def multiple_processing_func(input_data_series, input_date, train_window = 12):
    # global fitted_models
    # global processed_news_dict
    if input_date - train_window < min(input_data_series.index):
        return
    else:
        model = FourGramModel(input_data_series, input_date, train_window )
        try:
            model()
        except Exception as e:
            print('\033[31m')
            print("Training failed;" + e)
            print('\033[0m')
    #print("Dumping Pickle file.........\n")
    result_dir = "D:\\GIC_Project\\Tingjun\\data\\2_models\\bank\\"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    pik_name = str(input_date) + '_' + str(train_window)
    pickle_path = result_dir + pik_name + '.pickle'
    pickle_out = open(pickle_path, "wb")
    pickle.dump(model.lm, pickle_out)
    pickle_out.close()
    print("Finish dumping" + pik_name + "\n")


def multi_train_handler(data_series, frequency = 'M'):

    p = mp.Pool()
    if frequency == 'M':
        date_list = [np.datetime64(key).astype('datetime64[M]') for key in data_series.index]
    else:
        date_list = [np.datetime64(key).astype('datetime64[D]') for key in data_series.index]
    train_model = partial(multiple_processing_func, data_series)
    p.map_async(train_model, date_list)
    p.close()
    p.join()


def run(frequency = 'M'):
    if not os.path.exists(model_dir):
        print("\ncreating dir: ", model_dir, "\n")
        os.makedirs(model_dir)

    candidates_tech = ['Amazon', 'Apple','Facebook','Google','IBM','Intel','Lyft','Microsoft','Netflix',
                       'Qualcomm','Snap','Tesla','Uber']

    candidates_bank = ['Bank of America', 'Barclays', 'Citigroup', 'Credit Suisse', 'Deutsche Bank', 'Goldman Sachs',
                       'HSBC','JPMorgan Chase', 'Morgan Stanley', 'UBS', 'Wells Fargo']
    temp_series = []
    for company in candidates_bank:

        dict_pickle_path = "../data/cleaned/{}_{}.pickle".format(company,frequency)
        pickle_in = open(dict_pickle_path, "rb")
        temp_series.append( pickle.load(pickle_in))
        pickle_in.close()
    df = pd.concat(temp_series,axis = 1)
    df = df.apply(lambda s: s.fillna({i: [] for i in df.index}))
    processed_news_series = df.apply(lambda x: x.sum(), axis=1)

    start = time.time()
    multi_train_handler(processed_news_series)
    end = time.time()
    print("All models fitted, total Time:" + str(end - start))

if __name__ == '__main__':
    run()


