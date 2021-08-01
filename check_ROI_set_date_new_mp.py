import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import multiprocessing as mp
from multiprocessing.pool import Pool
from functools import partial
# import matplotlib as plt

# 첫 신호 발생 날로만 기준잡음
#TODO 각 종목별 종가 기록 그래프
#TODO 코인은 장기로 봤을 때 안걸림

# siganl_data_3099 = pd.read_pickle('C:/Users/soso6/Documents/GitHub/Algorithm trading/result/result_3099.pickle')
coin_path = 'C:\\Users\\soso6\\Documents\\GitHub\\Algorithm trading\\result\\Strong_complete\\coin\\'
stock_path = 'C:\\Users\\soso6\\Documents\\GitHub\\Algorithm trading\\result\\Strong_complete\\stock\\'
coin_file = os.listdir(coin_path)
stock_file = os.listdir(stock_path)
# signal_data_3099 = pd.read_pickle('result_3099.pickle')
# signal_data_3099.dropna(axis=1, thresh=1.00, inplace=True) # 구매신호가 발생하지 않은 종목 drop



def ROI(initial, end):
    return round((((end/initial)-1) * 100), 2)

def get_outlier(df=None, column=None, weight=1.5):
    # target 값과 상관관계가 높은 열을 우선적으로 진행
    quantile_25 = np.percentile(df[column].values, 25)
    quantile_75 = np.percentile(df[column].values, 75)

    IQR = quantile_75 - quantile_25
    IQR_weight = IQR*weight

    lowest = quantile_25 - IQR_weight
    highest = quantile_75 + IQR_weight

    outlier_idx = df[column][ (df[column] < lowest) | (df[column] > highest) ].index
    return outlier_idx

def remove_outlier(df=None, column=None, weight=1.5):
    # target 값과 상관관계가 높은 열을 우선적으로 진행
    quantile_25 = np.percentile(df[column].values, 25)
    quantile_75 = np.percentile(df[column].values, 75)

    IQR = quantile_75 - quantile_25
    IQR_weight = IQR*weight

    lowest = quantile_25 - IQR_weight
    highest = quantile_75 + IQR_weight

    outlier_idx = df[column][ (df[column] < lowest) | (df[column] > highest) ].index
    return outlier_idx

start_date = '2019-01-01'
end_date = '2019-12-31'
upper_range = [x for x in range(5,40)]
lower_range = [x for x in range(-40,-5)]
holding_range = [x for x in range(15,80)]

for upper in tqdm(upper_range):
    for lower in lower_range:
        for holding in holding_range:
            max_value = {}
            max_value_day = {}
            min_value = {}
            min_value_day = {}
            roi = {}
            roi_day = {}
            # holding_day = 60 # TODO 보유일 수 설정
            # start_date = '2019-01-01'
            # end_date = '2019-12-31'
            # upper_ROI = 15
            # lower_ROI = -7
            holding_day = holding # TODO 보유일 수 설정
            upper_ROI = upper
            lower_ROI = lower



            count = 0


            for file in stock_file:
                start_idx = None
                data = pd.read_pickle(stock_path+file)
                data.set_index('date', inplace=True)
                data = data[start_date:end_date].copy()
                data.dropna(inplace=True, thresh=1, axis=1)
                if len(data.columns) in (0, 1): # 구매신호 없는 종목은 skip
                    continue

                start_count = 0
                for i, idx in enumerate(data.index):
                    val = data._get_value(idx, data.columns[0])

                    if val != None:
                        start_idx = i #해당 연도 내에 첫 구매신호만 포착
                        start_count += 1 # TODO 이러면 연속이 아니구나
                        if start_count == 5:
                            break

                if start_count != 5:
                    continue


                if start_idx == None:
                    continue
                elif len(data[start_idx:]) < holding_day: #남은 기간이 부족하면 구매x
                    continue
                else:
                    count += 1

                    data = data.iloc[start_idx:start_idx+holding_day].copy()

                    buy_price = data.iloc[0,1]
                    roi_day[file] = []
                    for p_idx in data.index:
                        now_price = data._get_value(p_idx, data.columns[1])
                        roi_now = ROI(buy_price, now_price)
                        roi_day[file].append(roi_now)
                        if roi_now < lower_ROI or roi_now > upper_ROI:
                            roi[file] = roi_now

                            break
                        elif p_idx == data.index[-1]:

                            roi[file] = roi_now



                # for i in data_origin.index:
                #     val = data_origin._get_value(i, 'signal')
                #     if type(val) == int:
                #         if int(col) == val:
                #             start_idx = i
                #
                #             break
                #     else:
                #         val = val.split('_')
                #         if str(col) in val:
                #             start_idx = i
                #
                #         break
                #
                # if start_idx == None:
                #     continue
                #
                # elif len(data_origin.loc[start_idx:start_idx+holding_day]) < holding_day:
                #     # 구매신호 발생 후 종가 데이터 수가 목표 보유 일 수보다 적은 종목 pass
                #     continue
                # else:
                #     count += 1 # 테스트 종목 수 카운트
                #     data = data_origin.loc[start_idx:start_idx+holding_day, col].reset_index(drop=True)
                #     initial = data.iloc[0] # 구매 신호 발생 시 종가 (구매가)
                #
                #     # 구매 후 설정 보유기간 동안 최대 수익률 및 해당 일 수
                #     end = data.max()
                #
                #     end_idx = data.index[data == end][0]
                #     max_value[col] = ROI(initial, end)
                #     max_value_day[col] = end_idx + 1
                #
                #     # 구매 후 설정 보유기간 동안 최대 손실률 및 해당 일 수
                #     end = data.min()
                #     end_idx = data.index[data == end][0]
                #     min_value[col] = ROI(initial, end)
                #     min_value_day[col] = end_idx + 1
                #
                #     # 구매 후 설정 보유기간 도달 시 가격
                #     end = data.iloc[holding_day-1]
                #
                #     roi[col] = ROI(initial, end)

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
                max_result.append(np.max(roi_day[key]))
                min_result.append(np.min(roi_day[key]))


            print('일자:',start_date,'~',end_date)
            print('테스트 종목 수:', count,'/',len(stock_file))
            print()
            print('보유 일(봉) 수:', holding_day)
            print('익절선:',upper_ROI)
            print('손절선:',lower_ROI)
            print()
            # print('평균 최대 수익:',round(np.average(max_result),2)) # 평균
            # print('평균 최대 수익 평균 발생 일 수:', round(np.average(max_day_result),2))
            # print('평균 최대 손실:',round(np.average(min_result),2))
            # print('평균 최대 손실 평균 발생 일 수:',round(np.average(min_day_result),2))
            print('평균 보유 수익률:',round(np.average(holding_result),2))

            # print('최대 수익률', max_value)
            # print('최대 손실률', min_value)
            # print('무지성 보유일 손익률', roi)