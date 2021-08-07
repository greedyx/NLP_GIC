from GoogleNews import GoogleNews
from newspaper import Article
from newspaper import Config
import pandas as pd
import nltk
import time


#config will allow us to access the specified url for which we are #not authorized. Sometimes we may get 403 client error while parsing #the link to download the article.
nltk.download('punkt')

user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'
config = Config()
config.browser_user_agent = user_agent
config.request_timeout = 10


googlenews=GoogleNews(start='05/01/2020',end='05/31/2020')
googlenews.search('Amazon')
result=googlenews.result()
df=pd.DataFrame(result)

for i in range(2,20):
    googlenews.getpage(i)
    result=googlenews.result()
    df=pd.DataFrame(result)
list=[]

start = time.time()
for ind in df.index:
    dict={}
    article = Article(df['link'][ind],config=config)
    try:
        article.download()
        article.parse()
        article.nlp()
        dict['Date']=df['date'][ind]
        dict['Media']=df['media'][ind]
        dict['Title']=article.title
        dict['Article']=article.text
        dict['Summary']=article.summary
        list.append(dict)
    except:
        continue
news_df=pd.DataFrame(list)
end = time.time()


print("start='05/01/2020',end='05/31/2020' , Cost time " + str(end - start) + "\n")

print('Total news we have' + str(len(df)))
print('Total news we can download' + str(len(news_df)))