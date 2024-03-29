import os

import numpy as np
import pandas as pd
from tqdm import tqdm

# import matplotlib as plt

# 첫 신호 발생 날로만 기준잡음
# TODO 각 종목별 종가 기록 그래프
# TODO 코인은 장기로 봤을 때 안걸림

# siganl_data_3099 = pd.read_pickle('C:/Users/soso6/Documents/GitHub/Algorithm trading/result/result_3099.pickle')
coin_path = 'C:\\Users\\soso6\\Documents\\GitHub\\Algorithm trading\\result\\Strong_short\\coin\\'
stock_path = 'C:\\Users\\soso6\\Documents\\GitHub\\Algorithm trading\\result\\Strong_complete_0802\\stock\\'
item_path = 'C:\\Users\\soso6\\Documents\\GitHub\\Algorithm trading\\stock_data\\CYBOS\\default\\stockitems_analysis.pickle'
item = pd.read_pickle(item_path)
coin_file = os.listdir(coin_path)
stock_file = os.listdir(stock_path)


# signal_data_3099 = pd.read_pickle('result_3099.pickle')
# signal_data_3099.dropna(axis=1, thresh=1.00, inplace=True) # 구매신호가 발생하지 않은 종목 drop


def ROI(initial, end):
    return round((((end / initial) - 1) * 100), 2)


def get_outlier(df=None, column=None, weight=1.5):
    # target 값과 상관관계가 높은 열을 우선적으로 진행
    quantile_25 = np.percentile(df[column].values, 25)
    quantile_75 = np.percentile(df[column].values, 75)

    IQR = quantile_75 - quantile_25
    IQR_weight = IQR * weight

    lowest = quantile_25 - IQR_weight
    highest = quantile_75 + IQR_weight

    outlier_idx = df[column][(df[column] < lowest) | (df[column] > highest)].index
    return outlier_idx


def remove_outlier(df=None, column=None, weight=1.5):
    # target 값과 상관관계가 높은 열을 우선적으로 진행
    quantile_25 = np.percentile(df[column].values, 25)
    quantile_75 = np.percentile(df[column].values, 75)

    IQR = quantile_75 - quantile_25
    IQR_weight = IQR * weight

    lowest = quantile_25 - IQR_weight
    highest = quantile_75 + IQR_weight

    outlier_idx = df[column][(df[column] < lowest) | (df[column] > highest)].index
    return outlier_idx


max_value = {}
max_value_day = {}
min_value = {}
min_value_day = {}
roi = {}
roi_day_high = {}
roi_day_low = {}
date_buy = {}
is_G_B = {}
stock_name = {}
holding_day = 20  # 보유일 수 설정
start_date = '2019-01-01'
end_date = '2019-12-31'
upper_ROI = 20  # 익절선
lower_ROI = -15  # 손절선
save = True
filename = 'Strong_종목'

filename += '.csv'
count = 0

for file in tqdm(stock_file):
    start_idx = None
    data = pd.read_pickle(stock_path + file)
    data.set_index('date', inplace=True)
    data = data[start_date:end_date].copy()

    data.dropna(inplace=True, thresh=1, axis=1)
    if len(data.columns) in (0, 8):  # 구매신호 없는 종목은 skip
        continue

    idx_count = []
    for i, idx in enumerate(data.index):
        val = data._get_value(idx, data.columns[0])  # 시그널 체크

        if val != None:
            start_idx = i  # 해당 연도 내에 첫 구매신호만 포착
            idx_count.append(1)

            if len(idx_count) == 3:
                break
        else:
            idx_count = []
    #
    if len(idx_count) != 3:
        continue

    if start_idx == None:
        continue
    elif len(data[start_idx:]) < holding_day:  # 남은 기간이 부족하면 구매x
        continue
    else:
        count += 1
        date_buy[file] = idx

        data = data.iloc[start_idx:start_idx + holding_day].copy()

        buy_price = data._get_value(data.index[0], data.columns[1])

        continuous_sma = []
        watch_loss = []
        stock_name[file] = item[item['code'] == file[:-7]]['name'].iloc[0]
        for num ,p_idx in enumerate(data.index):  # 익절선, 손절선, 최종 보유일 이후 테스트
            now_close = data._get_value(p_idx, file[:-7])
            now_high = data._get_value(p_idx, 'high')
            now_low = data._get_value(p_idx, 'low')
            now_sma_w_20 = data._get_value(p_idx, 'sma_w_20')


            # 손절선 익절선
            roi_now_high = ROI(buy_price, now_high)
            roi_now_low = ROI(buy_price, now_low)

            if p_idx == data.index[-1]: # 보유기간 지남
                roi[file] = ROI(buy_price, now_close)
                is_G_B[file] = '보유기간'
                break

            if roi_now_high >= upper_ROI:
                roi[file] = upper_ROI
                is_G_B[file] = '익절'
                break


            if data.loc[data.index[num+1],'sma_w_20'] != data.loc[data.index[num],'sma_w_20']: # 주의 마지막 날
                if now_low < now_sma_w_20:
                    watch_loss.append(1)
                else:
                    watch_loss = []
            if watch_loss == 2:
                roi[file] = ROI(buy_price, now_low)
                is_G_B[file] = "손절"
                break

            if roi_now_low < lower_ROI:
                roi[file] = lower_ROI
                is_G_B[file] = "손절선"
                break

        roi_day_high[file] = []
        roi_day_low[file] = []
        for p_idx in data.index:  # 설정한 보유기간 내 최대 수익/손실
            now_close = data._get_value(p_idx, data.columns[1])
            now_high = data._get_value(p_idx, data.columns[2])
            now_low = data._get_value(p_idx, data.columns[3])

            roi_now_high = ROI(buy_price, now_close)
            roi_now_low = ROI(buy_price, now_close)
            roi_day_high[file].append(roi_now_high)
            roi_day_low[file].append(roi_now_low)

max_result = []
max_day_result = []
min_result = []
min_day_result = []
holding_result = []
for key in list(roi.keys()):
    # max_result.append(max_value[key])
    # max_day_result.append(max_value_day[key])
    # min_result.append(min_value[key])
    # min_day_result.append(min_value_day[key])
    holding_result.append(roi[key])
for key in list(roi.keys()):
    max_result.append(np.max(roi_day_high[key]))
    min_result.append(np.min(roi_day_low[key]))

print('일자:', start_date, '~', end_date)
print('테스트 종목 수:', count, '/', len(stock_file))
print()
print('보유 일(봉) 수:', holding_day)
print('익절선:', upper_ROI)
print('손절선:', lower_ROI)
print()
print('평균 최대 수익:', round(np.average(max_result), 2))  # 평균
# print('평균 최대 수익 평균 발생 일 수:', round(np.average(max_day_result),2))
print('평균 최대 손실:', round(np.average(min_result), 2))
# print('평균 최대 손실 평균 발생 일 수:',round(np.average(min_day_result),2))
print('평균 최종 수익률:', round(np.average(holding_result), 2))
print('누계 최종 수익률:', round(np.sum(holding_result), 2))

# print('최대 수익률', max_value)
# print('최대 손실률', min_value)
# print('무지성 보유일 손익률', roi)

# TOP 30 손실 종목 정리
YH_plus = []
YH_result = []
holding_result.sort()

for key, value in roi.items():
    YH_result.append([stock_name[key], value, date_buy[key], is_G_B[key]])

column = ['종목명','ROI','매도 날짜','판매 유형']
result = pd.DataFrame(YH_result, columns=column)
# plus_30 = pd.DataFrame(YH_plus, columns=column)
if save == True:
    result.to_csv(filename, encoding='euc_kr') #TODO 엑셀 파일 생성
else:
    pass
# plus_30.to_csv('테스트한 익절 손절 조건_plus.csv')