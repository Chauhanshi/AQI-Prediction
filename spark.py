#multi-threading, thread pool concept to pull the data from url
from sqlalchemy import create_engine
from multiprocessing.pool import ThreadPool
tpool = ThreadPool(processes=5)

#I have used the spark in order to store the data into to the postgres, this also enables to distributed processing
from pyspark.sql import SparkSession
spark = SparkSession.builder.master("local[*]").getOrCreate()



import os
import numpy as np
import pandas as pd
import json
import re
import random
import itertools
import urllib.request, json
with urllib.request.urlopen("https://healthdata.gov/data.json") as url:
    Health_data = json.loads(url.read().decode())


data_keywords= []
for i in Health_data['dataset']:
    keyword = i['keyword']
    data_keywords.append(keyword)


flat_list = list(itertools.chain(*data_keywords))
df_keywords = pd.DataFrame(flat_list)
covid_keys = df_keywords.loc[(df_keywords[0].str.contains('covid') == True) | (df_keywords[0].str.contains('coronav')  == True)].drop_duplicates()


Key_Download = pd.DataFrame(columns = ['Keyword','downloadURL'])
for i in range(len(Health_data['dataset'])):
    keyword = Health_data['dataset'][i]['keyword']
    download_url = Health_data['dataset'][i].get('distribution', 'Not Found')
    if(download_url !='Not Found'):
        if "downloadURL" in download_url[0]:
            download_url = download_url[0]['downloadURL']
        elif ("accessURL" in download_url[0] and "downloadURL" not in download_url[0]):
            download_url = download_url[0]['accessURL']

    check =   any(item in keyword for item in covid_keys[0].values.tolist())
    if (check == True):
        Key_Download = Key_Download.append({'Keyword':keyword, 'downloadURL': str(download_url)}, ignore_index= True)

Key_Download.drop(Key_Download[Key_Download['downloadURL'] == ('Not Found')].index, inplace = True)
url_df = Key_Download["downloadURL"]

l_url = []
for index, url in url_df.items():
    d_url = {'url': url, 'index' : index}
    l_url.append(d_url)

def urlthread(task):  
        url = task['url']
        index = task['index']
        urllib.request.urlretrieve(url, "cdcData" + format(index) + ".csv")
        
l_e = tpool.map(urlthread, l_url)
#print(l_e)

#class Config:
 #   username = 'postgres'
  #  password = 'Viggy.02'
   # db = 'postgres'

#engine = create_engine('postgresql://'+Config.username+':'+Config.password+'@localhost:5432/'+Config.db)
#folder = os.path.join(os.getcwd(), 'Different states')

covid = spark.read.csv('cdcData8.csv',header=True)

#for State in covid['State'].unique():
# Filter the dataframe using that column and value from the list
# state_df = covid[covid['State']==State]
#    state_df.to_sql(State, engine, if_exists='replace')
covid.registerTempTable("covid")
state_df = spark.sql("select distinct State from covid")
#state_df = state_df.replace(' ', '_')
state_df.show()

for state in state_df.rdd.collect():
    state_str = state["State"]
    #print(state_str)
    covid_state_df = covid.filter(covid['State'] == state_str)
    state_str = state_str.replace(' ', '_')
    #print(state_str)
    table_name = 'public.' +  state_str
    #print(table_name)
    covid_state_df.write \
    .mode("overwrite") \
    .format("jdbc") \
    .option("url", "jdbc:postgresql://localhost:5432/postgres") \
    .option("dbtable", table_name) \
    .option("user", "postgres") \
    .option("password", "Viggy.02") \
    .option("driver", "org.postgresql.Driver") \
    .save()


