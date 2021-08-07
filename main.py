import sys
sys.path.append('D:\GIC_Project\Tingjun')
import pickle
import pandas as pd
import os
from data_clean.data_cleaner import DataCleaner
from param_calc.freq_token import  FreqToken

#from param_calc.entropy_calc import Entropy


# 1. data clean,需要修改的地方（主要是路径）, 会有comment在该行末端
company = 'Amazon'  # 目标公司
cleaned_dir = "D:/GIC_Project/Tingjun/data/cleaned/"  # data clean之后的输出file所在文件夹，会在下面自动创建
if not os.path.exists(cleaned_dir):
    os.makedirs(cleaned_dir)

# clean data1
def data_clean1(company,frequency:str = 'M'):
    # frequency choose 'M' or 'D'
    series = pd.read_csv(r"D:\GIC_Project\Tingjun\data\{}.csv".format(company))  # 1.你自己的某个公司的csv文件所在位置
    cleaner = DataCleaner(series,frequency)
    cleaner()
    series_cleaned = cleaner.m_data_series

    # Put cleaned data into a pickle, company by company
    if frequency == 'M':
        pickle_out = open(cleaned_dir + "{}_M.pickle".format(company), "wb")
    else:
        pickle_out = open(cleaned_dir + "{}_D.pickle".format(company), "wb")

    pickle.dump(series_cleaned, pickle_out)
    pickle_out.close()
    return series_cleaned

candidates_tech = ['Amazon', 'Apple','Facebook','Google','IBM','Intel','Lyft','Microsoft','Netflix',
                       'Qualcomm','Snap','Tesla','Uber']

candidates_bank = ['Bank of America', 'Barclays', 'Citigroup', 'Credit Suisse', 'Deutsche Bank','Goldman Sachs', 'HSBC',
                   'JPMorgan Chase', 'Morgan Stanley', 'UBS', 'Wells Fargo']

for company in candidates_bank:
    series_cleaned = data_clean1(company,'M')
test = series_cleaned.iloc[:2]
# !! 如果只是要做text clean的話，其實只运行前面的程序，然后用 series_cleand 里的text数据就好，然后一个一个排除一些混乱的语句
# 具体修改可以查看 data_clean.data_cleaner 的 DataCleaner 类，你可以在里面直接修改，也可以把你零散修改的直接发给我



# 2. freq_token中生成 3grams ,为第4步做准备
grams3_dir = "D:/GIC_Project/Tingjun/data/3grams/"  # 3grams data输出文件所在文件夹
if not os.path.exists(grams3_dir):
    os.makedirs(grams3_dir)

pickle_in = open(cleaned_dir + "{}.pickle".format(company), "rb")
cleaned_news = pickle.load(pickle_in)
pickle_in.close()

freToken = FreqToken(test)
freToken()

test_3grams = freToken.frequent_grams

pickle_out = open(grams3_dir + "{}_3grams.pickle".format(company), "wb")
pickle.dump(freToken.frequent_grams, pickle_out)
pickle_out.close()
print("Finished dumping " + company + "/3_freq_grams" +  ".pickle\n")

gram3_model = freToken.frequent_grams
first_key = list(gram3_model.keys())[0]
print(gram3_model[first_key])  # 某一个月的，4-gram 模型的字典，会显示4gram出现的频率
# 我又觉得这里好正常。。。。



# 3.train and store model, 生成 2_models，为第4步做准备
# ! 这一步骤会有点卡，因为用到多线程
# 无法直接放在main里，不然运行会报错，运行完前两步后去  fourgram.train_and_store_model 修改参数并运行
model2_dir = 'D:/GIC_Project/Tingjun/data/2_models'

grams3_dir = "D:/GIC_Project/Tingjun/data/3grams/"
pickle_in = open(grams3_dir + "{}_3grams.pickle".format(company), "rb")
freq_ngrams = pickle.load(pickle_in)
pickle_in.close()

# 4. 最后，entropy和sentiment计算
#这里先不管好了。。。也得在  param_calc.entropy_calc 里修改参数并运行

