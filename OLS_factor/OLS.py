import yfinance as yf
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import numpy as np


company = 'Amazon'
path = r'D:\GIC_project\Tingjun\data\regress_data\{}\{}_x.csv'.format(company,company)
data = pd.read_csv(path).iloc[:,1:]
data = data.set_index('date')


def yahoo(ticker):
    # nasdaq 100 technology sector index
    price = yf.download(ticker,
                        start='2016-01-01',
                        end='2020-12-31',
                        progress=False)
    price_d = np.log(price / price.shift(1)).dropna()
    price_vol = price_d['Adj Close'].resample('M').std()

    price_volume = price['Volume'].resample('M').sum()
    price_m = price.resample('M').last()
    price_m['Volume'] = price_volume
    r_price_m = np.log(price_m/price_m.shift(1)).dropna()
    r_price_m['vol'] = price_vol
    r_price_m['date'] = r_price_m.index
    r_price_m['date'] = r_price_m['date'].apply(lambda x:x.replace(day = 1))
    def date2str(a):
        return a.strftime("%Y-%m-%d")
    r_price_m['date'] = r_price_m['date'].apply(lambda x:date2str(x))
    r_price_m = r_price_m.set_index('date')
    return r_price_m

def yahoo_index(ticker):
    # nasdaq 100 technology sector index
    price = yf.download(ticker,
                        start='2016-01-01',
                        end='2020-12-31',
                        progress=False)
    price = price.drop(['Volume'],axis = 1)
    price_d = np.log(price / price.shift(1)).dropna()
    price_vol = price_d['Adj Close'].resample('M').std()


    price_m = price.resample('M').last()
    r_price_m = np.log(price_m/price_m.shift(1)).dropna()
    r_price_m['vol'] = price_vol
    r_price_m['date'] = r_price_m.index
    r_price_m['date'] = r_price_m['date'].apply(lambda x:x.replace(day = 1))
    def date2str(a):
        return a.strftime("%Y-%m-%d")
    r_price_m['date'] = r_price_m['date'].apply(lambda x:date2str(x))
    r_price_m = r_price_m.set_index('date')
    return r_price_m

r_price_m = yahoo(ticker = 'AMZN')

df_ols = data.join(r_price_m).drop(['Open','Close','High','Low'],axis = 1)

def sent_plot(df_ols):
    plt.figure(figsize=(8,6))
    plt.subplot(231)
    plt.plot(df_ols['Adj Close'].values)
    plt.title('return')
    plt.tight_layout()

    plt.subplot(232)
    plt.plot(df_ols['Volume'].values)
    plt.title('volume change percentage')
    plt.tight_layout()

    plt.subplot(233)
    plt.plot(df_ols['vol'].values)
    plt.title('volatility')
    plt.tight_layout()


    plt.subplot(234)
    plt.plot(df_ols['entpos'].values,label = 'positive')
    plt.plot(df_ols['entneg'].values,label = 'negative')
    plt.title('entropy')
    plt.legend()
    plt.tight_layout()

    plt.subplot(235)
    plt.plot(df_ols['sentpos'].values,label = 'positive')
    plt.plot(df_ols['sentneg'].values,label = 'negative')
    plt.title('sentiment')
    plt.legend()
    plt.tight_layout()

    plt.subplot(236)
    plt.plot(df_ols['entsent_pos'].values,label = 'positive')
    plt.plot(df_ols['entsent_neg'].values,label = 'negative')
    plt.title('entropy*sentiment ')
    plt.tight_layout()
    plt.legend()
    return


def sent_plot_index(df_ols):
    plt.figure(figsize=(8,6))
    plt.subplot(231)
    plt.plot(df_ols['Adj Close'].values)
    plt.title('return')
    plt.tight_layout()


    plt.subplot(232)
    plt.plot(df_ols['vol'].values)
    plt.title('volatility')
    plt.tight_layout()


    plt.subplot(234)
    plt.plot(df_ols['entpos'].values,label = 'positive')
    plt.plot(df_ols['entneg'].values,label = 'negative')
    plt.title('entropy')
    plt.legend()
    plt.tight_layout()

    plt.subplot(235)
    plt.plot(df_ols['sentpos'].values,label = 'positive')
    plt.plot(df_ols['sentneg'].values,label = 'negative')
    plt.title('sentiment')
    plt.legend()
    plt.tight_layout()

    plt.subplot(236)
    plt.plot(df_ols['entsent_pos'].values,label = 'positive')
    plt.plot(df_ols['entsent_neg'].values,label = 'negative')
    plt.title('entropy*sentiment ')
    plt.tight_layout()
    plt.legend()
    return

company = 'Morgan Stanley'
path = r'D:\GIC_project\Tingjun\data\regress_data\bank\{}\{}_x.csv'.format(company,company)
data = pd.read_csv(path).iloc[:,1:]
data = data.set_index('date')

r_price_ms = yahoo('MS')
df_ols = data.join(r_price_ms).drop(['Open','Close','High','Low'],axis = 1)
sent_plot(df_ols)





# nasdaq 100 technology sector index
r_price_tech = yahoo_index('^NDXT')

candidates_tech = ['Amazon', 'Apple','Facebook','Google','IBM','Intel','Lyft','Microsoft','Netflix',
                       'Qualcomm','Snap','Tesla','Uber']

candidates_bank = ['Bank of America', 'Barclays', 'Citigroup', 'Credit Suisse', 'Deutsche Bank','Goldman Sachs', 'HSBC',
                   'JPMorgan Chase', 'Morgan Stanley', 'UBS', 'Wells Fargo']


df_total_list = []
#df_total = pd.DataFrame(columns=['entpos', 'sentpos', 'entsent_pos', 'entneg', 'sentneg', 'entsent_neg'])
for company in candidates_tech:
    path = r'D:\GIC_project\Tingjun\data\regress_data\{}\{}_x.csv'.format(company,company)
    data = pd.read_csv(path).iloc[:,1:]
    data =data.set_index('date')
    df_total_list.append(data)

df_total = pd.concat(df_total_list )



df_aggregate = pd.DataFrame(index = data.index,
    columns=['entpos', 'sentpos', 'entsent_pos', 'entneg', 'sentneg', 'entsent_neg'])
for idx in df_aggregate.index:
    temp = df_total.loc[idx]
    df_aggregate.loc[idx] = temp.mean()
df_aggregate = df_aggregate.round(5)

df_ols_tech = df_aggregate.join(r_price_tech).drop(['Open','Close','High','Low'],axis = 1)

sent_plot_index(df_ols_tech)


# Nasdaq Bank
bank = yf.download('^BANK',
 start='2016-01-01',
 end='2020-12-31',
 progress=False)

