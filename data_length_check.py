import pandas as pd
import numpy as np
import os
from tqdm import tqdm

coin_path = 'C:\\Users\\soso6\\Documents\\GitHub\\Algorithm trading\\coin_data\\test\\days\\'
stock_path = 'C:\\Users\\soso6\\Documents\\GitHub\\Algorithm trading\\stock_data\\CYBOS\\indicator\\Strong(complete)\\daily_total.pickle'
item_path = 'C:\\Users\\soso6\\Documents\\GitHub\\Algorithm trading\\stock_data\\CYBOS\\default\\stockitems_analysis.pickle'
file_list = os.listdir(coin_path)

length_coin = []
i_date_coin = []

for file in file_list:
    data = pd.read_csv(coin_path + file)
    length_coin.append([len(data)])
    i_date_coin.append(data['time'][0])



len_coin = pd.DataFrame(length_coin)
i_date_coin = pd.DataFrame(i_date_coin)


coin_length = len_coin.value_counts()
i_coin_date = i_date_coin.value_counts()


stock = pd.read_pickle(stock_path)
item = pd.read_pickle(item_path)
length_stock = []
i_date_stock = []

for code in tqdm(item['code']):
    pp = stock[stock['code'] == code].reset_index(drop=True)
    length_stock.append(len(pp))
    i_date_stock.append(pp['date'][0])


result_stock = pd.DataFrame(length_stock)
i_date_stock = pd.DataFrame(i_date_stock)


stock_length = result_stock.value_counts()
i_coin_stock = i_date_stock.value_counts()
