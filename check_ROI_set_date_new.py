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
stock_path = 'C:\\Users\\soso6\\Documents\\GitHub\\Algorithm trading\\result\\Strong_complete_K\\stock\\'
item_path = 'C:\\Users\\soso6\\Documents\\GitHub\\Algorithm trading\\stock_data\\CYBOS\\default\\stockitems_analysis.pickle'
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
date_dict = {}
holding_day = 60  # 보유일 수 설정
start_date = '2019-01-01'
end_date = '2019-12-31'
upper_ROI = 20  # 익절선
lower_ROI = -5  # 손절선

count = 0

for file in tqdm(stock_file):
    start_idx = None
    data = pd.read_pickle(stock_path + file)
    data.set_index('date', inplace=True)
    data = data[start_date:end_date].copy()
    data.dropna(inplace=True, thresh=1, axis=1)
    if len(data.columns) in (0, 7):  # 구매신호 없는 종목은 skip
        continue

    idx_count = []
    for i, idx in enumerate(data.index):
        val = data._get_value(idx, data.columns[0])  # 시그널 체크

        if val != None:
            start_idx = i  # 해당 연도 내에 첫 구매신호만 포착
            idx_count.append(1)

            if len(idx_count) == 3 and idx_count[-1] == 1:
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
        date_dict[file] = idx
        data = data.iloc[start_idx:start_idx + holding_day].copy()

        buy_price = data._get_value(data.index[0], data.columns[1])

        continuous_sma = []

        for p_idx in data.index:  # 익절선, 손절선, 최종 보유일 이후 테스트
            now_close = data._get_value(p_idx, data.columns[1])
            now_high = data._get_value(p_idx, data.columns[2])
            now_low = data._get_value(p_idx, data.columns[3])
            now_sma_448 = data._get_value(p_idx, data.columns[4])
            now_sma_120 = data._get_value(p_idx, data.columns[5])
            now_sma_224 = data._get_value(p_idx, data.columns[6])
            now_sma_20 = data._get_value(p_idx, data.columns[7])

            # 손절선 익절선
            # roi_now_high = ROI(buy_price, now_high)
            roi_now_low = ROI(buy_price, now_low)

            # if roi_now_low <= lower_ROI:
            #     roi[file] = lower_ROI
            #     print('손절', lower_ROI)
            #     break
            # elif roi_now_high >= upper_ROI:
            #     roi[file] = upper_ROI
            #     break
            if now_high >= now_sma_448*1.05: # 익절
                roi[file] = ROI(buy_price, now_sma_448)
                print('익절', ROI(buy_price, now_sma_448))
                break
            elif p_idx == data.index[-1]: # 보유기간 지남
                print('보유기간 지남', ROI(buy_price, now_close))
                roi[file] = ROI(buy_price, now_close)

            if now_low <= now_sma_20: # 손절 (4일 연속 조건 count)
                continuous_sma.append(1)
            elif now_low > now_sma_20:
                continuous_sma = []

            if continuous_sma == [] : # 손절 (4일 연속인지 check)
                continue
            if len(continuous_sma) == 3 and continuous_sma[-1] == 1:
                roi[file] = ROI(buy_price, now_sma_20)
                print('손절', ROI(buy_price, now_sma_20))
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
# print('익절선:', upper_ROI)
# print('손절선:', lower_ROI)
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
YH_minus = []
item = pd.read_pickle(item_path)
holding_result.sort()
plus_top_30 = holding_result[-30:]
minus_top_30 = holding_result[:30]

# for key, value in roi.items():
#     for v in plus_top_30:
#         if v == value:
#             name = item[item['code'] == key[:-7]]['name'].iloc[0]
#             YH_plus.append([name, v, date_dict[key]])
#
# for key, value in roi.items():
#     for v in minus_top_30:
#         if v == value:
#             name = item[item['code'] == key[:-7]]['name'].iloc[0]
#             YH_minus.append([name, v, date_dict[key]])

for key, value in roi.items():
    for v in minus_top_30:
        if v == value:
            name = item[item['code'] == key[:-7]]['name'].iloc[0]
            YH_minus.append([name, v, date_dict[key]])

column = ['종목명','ROI','매도 날짜']
minus_30 = pd.DataFrame(YH_minus, columns=column)
plus_30 = pd.DataFrame(YH_plus, columns=column)

# minus_30.to_csv('테스트한 익절 손절 조건_minus.csv') #TODO 엑셀 파일 생성
# plus_30.to_csv('테스트한 익절 손절 조건_plus.csv')