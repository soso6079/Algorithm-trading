import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import matplotlib as plt

# 첫 신호 발생 날로만 기준잡음
#TODO 각 종목별 종가 기록 그래프
#TODO 코인은 장기로 봤을 때 안걸림

siganl_data_3099 = pd.read_pickle('C:/Users/soso6/Documents/GitHub/Algorithm trading/result/result_3099.pickle')
coin_path = 'C:\\Users\\soso6\\Documents\\GitHub\\Algorithm trading\\result\\Strong_complete\\coin\\'
stock_path = 'C:\\Users\\soso6\\Documents\\GitHub\\Algorithm trading\\result\\Strong_complete\\stock\\'
coin_file = os.listdir(coin_path)
stock_file = os.listdir(stock_path)
signal_data_3099 = pd.read_pickle('result_3099.pickle')
signal_data_3099.dropna(axis=1, thresh=1.00, inplace=True) # 구매신호가 발생하지 않은 종목 drop


signal_data_3099 = signal_data_3099.iloc[1906:2092, :]
siganl_data_3099 = siganl_data_3099.iloc[:]

def ROI(initial, end):
    return round((((end/initial)-1) * 100), 2)

max_value = {}
max_value_day = {}
min_value = {}
min_value_day = {}
roi = {}
roi_day = {}
holding_day = 40 # TODO 보유일 수 설정

print('전체 종목 수:', len(signal_data_3099.columns[2:]))

count = 0
start_idx = None

for col in tqdm(signal_data_3099.columns[2:]):

    data_origin = signal_data_3099.loc[:, ['signal', col]].dropna() # 구매 신호 이후 데이터만 남김

    # first_sig_idx = data_origin[data_origin['signal'].str.contains(str(col))].index[0]
    # 120 종목 시그널에도 1이 포함되서 안될듯

    for i in data_origin.index:
        val = data_origin._get_value(i, 'signal')
        if type(val) == int:
            if int(col) == val:
                start_idx = i
                break

        else:
            val = val.split('_')
            if str(col) in val:
                start_idx = i

            break

    if start_idx == None:
        continue

    elif len(data_origin.loc[start_idx:start_idx+holding_day]) < holding_day:
        # 구매신호 발생 후 종가 데이터 수가 목표 보유 일 수보다 적은 종목 pass
        continue
    else:
        count += 1 # 테스트 종목 수 카운트
        data = data_origin.loc[start_idx:start_idx+holding_day, col].reset_index(drop=True)
        initial = data.iloc[0] # 구매 신호 발생 시 종가 (구매가)

        # 구매 후 설정 보유기간 동안 최대 수익률 및 해당 일 수
        end = data.max()

        end_idx = data.index[data == end][0]
        max_value[col] = ROI(initial, end)
        max_value_day[col] = end_idx + 1

        # 구매 후 설정 보유기간 동안 최대 손실률 및 해당 일 수
        end = data.min()
        end_idx = data.index[data == end][0]
        min_value[col] = ROI(initial, end)
        min_value_day[col] = end_idx + 1

        # 구매 후 설정 보유기간 도달 시 가격
        end = data.iloc[holding_day-1]

        roi[col] = ROI(initial, end)

max_result = []
max_day_result = []
min_result = []
min_day_result = []
holding_result = []
for key in list(max_value.keys()):
    max_result.append(max_value[key])
    max_day_result.append(max_value_day[key])
    min_result.append(min_value[key])
    min_day_result.append(min_value_day[key])
    holding_result.append(roi[key])

print('테스트 종목 수:', count)
print()
print('보유 일(봉) 수:', holding_day)
print()
print('평균 최대 수익:',round(np.average(max_result),2)) # 평균
print('평균 최대 수익 평균 발생 일 수:', round(np.average(max_day_result),2))
print('평균 최대 손실:',round(np.average(min_result),2))
print('평균 최대 손실 평균 발생 일 수:',round(np.average(min_day_result),2))
print('평균 보유 수익률:',round(np.average(holding_result),2))
# print('최대 수익률', max_value)
# print('최대 손실률', min_value)
# print('무지성 보유일 손익률', roi)