from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import datetime  # For datetime objects
import os.path  # To manage paths
import sys  # To find out the script name (in argv[0])
import pandas as pd
import numpy as np

# Import the backtrader platform
import backtrader as bt
from custom_indicator import *
import argparse
import copy
import time


# 매수 전략
class Strong_complete(bt.Strategy): # bt.Strategy를 상속한 class로 생성해야 함.


    def log(self, txt, dt=None):
        ''' Logging function fot this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))


    def __init__(self):
        # 주식 보유 일수 카운트용
        self.holding = {}
        self.stock_num = 0
        self.date_count = 0

        self.name_list = self.getdatanames()
        del self.name_list[1]

        # 보유 기간 동안 최대 손익률 계산용
        self.column = ['date','signal']
        self.default_col = len(self.column)
        self.record = []
        # self.record_date = self.record[0]
        for i in range(len(self.datas) // 3):
            # self.record.append([])
            self.column.append(str(i))

        self.ind_dict = {}

        for i in range(len(self.datas)): # 각 종목별 지표 계산
            num = str(i // 3)

            if i % 3 == 0:# daily data
                self.ind_dict[num+'_sto_daily_20'] = Stochastic_slow_daeshin(self.datas[i])
                self.ind_dict[num+'_sma_vol_daily_20'] = bt.ind.SMA(self.datas[i].volume, period=20)
                self.ind_dict[num+'_sma_daily_448'] = bt.ind.SMA(self.datas[i], period=448)
                self.ind_dict[num+'_sma_daily_120'] = bt.ind.SMA(self.datas[i], period=120)
                self.ind_dict[num+'_sma_daily_224'] = bt.ind.SMA(self.datas[i], period=224)
                self.ind_dict[num+'_sma_daily_20'] = bt.ind.SMA(self.datas[i], period=20)

            elif i % 3 == 1:# weekly data
                self.ind_dict[num+'_sto_weekly_20'] = Stochastic_slow_daeshin(self.datas[i])
                self.ind_dict[num+'_sto_weekly_12'] = Stochastic_slow_daeshin(self.datas[i],
                                                                              period=12,
                                                                              period_dfast=5,
                                                                              period_dslow=5)
                self.ind_dict[num+'_sto_weekly_5'] = Stochastic_slow_daeshin(self.datas[i],
                                                                             period=5,
                                                                             period_dfast=3,
                                                                             period_dslow=3)
                self.ind_dict[num+'_sto_weekly_18'] = Stochastic_slow_daeshin(self.datas[i],
                                                                             period=18,
                                                                             period_dfast=10,
                                                                             period_dslow=10)
                self.ind_dict[num+'_MACD_weekly'] = bt.ind.MACD(self.datas[i])
                self.ind_dict[num+'_sma_weekly_20'] = bt.ind.SMA(self.datas[i], period=20)

            elif i % 3 == 2:# monthly data
                self.ind_dict[num+'_sto_monthly_20'] = Stochastic_slow_daeshin(self.datas[i])
                self.ind_dict[num+'_sto_monthly_5'] = Stochastic_slow_daeshin(self.datas[i],
                                                                              period=5,
                                                                              period_dfast=3,
                                                                              period_dslow=3)
                self.ind_dict[num+'_MACD_monthly'] = bt.ind.MACD(self.datas[i])

    def next(self):

        # if self.holding.keys(): #보유하고 있는 주식이 있으면
        #     for i in self.holding.keys():
        #         self.holding[i] += 1 #보유 일수 증가

        # 최대 손익률 계산용 날짜 기록
        # self.record_date.append(self.datas[0].datetime.date(0).isoformat())
        self.record.append([])
        self.record[-1].append(self.datas[0].datetime.date(0).isoformat())
        self.record[-1].append(None)


        # if not self.position: # not in the market
        for i in range(len(self.datas)//3):
            data_idx = i * 3
            cond_A = self.ind_dict[str(i)+'_sto_weekly_20'].percK[0] > self.ind_dict[str(i)+'_sto_weekly_20'].percK[-1]
            cond_B = self.ind_dict[str(i)+'_sto_weekly_20'].percK[0] >= self.ind_dict[str(i)+'_sto_weekly_20'].percD[0]
            cond_C = self.ind_dict[str(i)+'_sto_weekly_5'].percK[0] <= 46
            cond_D = self.ind_dict[str(i)+'_sto_monthly_20'].percD[0] > self.ind_dict[str(i)+'_sto_monthly_20'].percD[-1]
            cond_E = self.ind_dict[str(i)+'_sto_monthly_5'].percD[0] > self.ind_dict[str(i)+'_sto_monthly_5'].percD[-1]
            cond_F = self.ind_dict[str(i)+'_MACD_monthly'].macd[0] > self.ind_dict[str(i)+'_MACD_monthly'].macd[-1]
            cond_G = self.ind_dict[str(i)+'_sto_weekly_12'].percK[0] <= 66
            cond_H = -900 <= self.ind_dict[str(i)+'_MACD_weekly'].macd[0] <= 500
            cond_I = self.ind_dict[str(i)+'_sto_weekly_20'].percK[0] <= 68
            cond_J = self.ind_dict[str(i)+'_sto_daily_20'].percK[0] <= 36
            cond_K = self.ind_dict[str(i)+'_sma_weekly_20'][0] < self.datas[data_idx+1].close[0] #20일 지지
            cond_L = self.ind_dict[str(i)+'_sto_weekly_18'].percK[0] > self.ind_dict[str(i)+'_sto_weekly_18'].percK[-1]
            cond_M = 100000 <= self.ind_dict[str(i)+'_sma_vol_daily_20'][0] <= 999999999

            if cond_A and cond_B and cond_C and cond_D and \
                    cond_E and cond_G and cond_H and cond_F and\
                    cond_I and cond_J and cond_K and cond_L and cond_M:
                # print('종목 번호',i)
                close = self.datas[data_idx].close[0] # 종가 값
                size = int(self.broker.getcash() / close) # 최대 구매 가능 개수
                # size = int(size/2)z
                size = 1


                self.buy(size=size) # 매수 size = 구매 개수 설정

                # self.log('BUY CREATE, %.2f' % self.datas[data_idx].close[0])

                # 보유기간 세기 위해
                self.holding[self.stock_num] = 0 #주식 구매 시 key 추가
                self.stock_num += 1

                # 보유기간 중 최대 손익률 검증용
                self.record[-1].append(self.datas[data_idx].close[0])
                self.record[-1].append(self.datas[data_idx].high[0])
                self.record[-1].append(self.datas[data_idx].low[0])
                self.record[-1].append(self.ind_dict[str(i)+'_sma_daily_448'][0])
                self.record[-1].append(self.ind_dict[str(i)+'_sma_daily_120'][0])
                self.record[-1].append(self.ind_dict[str(i)+'_sma_daily_224'][0])
                self.record[-1].append(self.ind_dict[str(i)+'_sma_daily_20'][0])

                if self.record[-1][1] == None:
                    self.record[-1][1] = str(self.name_list[i])
                else: #다른 종목도 시그널이 동일한 날짜에 발생했을 때
                    value = self.record[-1][1]
                    self.record[-1][1] = str(value)+'_'+str(self.name_list[i])


            else:#구매조건이 아닐 경우
                if self.date_count == 0:
                    for col in range(7):
                        self.record[0].append(None)

                elif self.record[-2][i+self.default_col] != None: # TODO 전날 구매했으면 종가 계속 기록

                    self.record[-1].append(self.datas[data_idx].close[0]) #종가 기록
                    self.record[-1].append(self.datas[data_idx].high[0]) #고가 기록
                    self.record[-1].append(self.datas[data_idx].low[0])  #저가 기록
                    self.record[-1].append(self.ind_dict[str(i)+'_sma_daily_448'][0])
                    self.record[-1].append(self.ind_dict[str(i)+'_sma_daily_120'][0])
                    self.record[-1].append(self.ind_dict[str(i)+'_sma_daily_224'][0])
                    self.record[-1].append(self.ind_dict[str(i)+'_sma_daily_20'][0])
                else: # 보유도 안하고 구매도 안했으면 None 기록
                    for col in range(7):
                        self.record[-1].append(None)

        # else: # not in position
        # if self.holding.keys():  #보유하고 있는 주식이 있으면
        #     key_list = copy.deepcopy(list(self.holding.keys()))
        #     for i in key_list:
        #         if self.holding[i] >= 60: #보유 일(봉)수
        #             self.close() # 매도
        #             # self.log('SELL CREATE, %.2f' % self.datas[i].close[0])
        #             del self.holding[i]

        # 최대 손익률 계산용 결과 생성
        # if self.record[-1][0] == '2021-07-13':
        #     result = pd.DataFrame(self.record, columns=self.column)
            # result.to_pickle('result/result_test.pickle')

        self.date_count += 1

def parse_args():
    parser = argparse.ArgumentParser(
        description='Pandas test script')

    parser.add_argument('--dataname', default='', required=False,
                        help='File Data to Load')

    parser.add_argument('--timeframe', default='weekly', required=False,
                        choices=['daily', 'weekly', 'monhtly'],
                        help='Timeframe to resample to')

    parser.add_argument('--compression', default=1, required=False, type=int,
                        help='Compress n bars into 1')

    return parser.parse_args()

class Strong_complete_short(bt.Strategy): # bt.Strategy를 상속한 class로 생성해야 함.


    def log(self, txt, dt=None):
        ''' Logging function fot this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))


    def __init__(self):
        # 주식 보유 일수 카운트용
        self.holding = {}
        self.stock_num = 0
        self.date_count = 0

        self.name_list = self.getdatanames()
        del self.name_list[1]

        # 보유 기간 동안 최대 손익률 계산용
        self.column = ['date','signal']
        self.default_col = len(self.column)
        self.record = []
        # self.record_date = self.record[0]
        for i in range(len(self.datas) // 3):
            # self.record.append([])
            self.column.append(str(i))

        self.ind_dict = {}

        for i in range(len(self.datas)): # 각 종목별 지표 계산
            num = str(i // 3)

            if i % 3 == 0:# daily data
                self.ind_dict[num+'_sto_daily_20'] = Stochastic_slow_daeshin(self.datas[i])
                self.ind_dict[num+'_sto_daily_12'] = Stochastic_slow_daeshin(self.datas[i],
                                                                              period=12,
                                                                              period_dfast=5,
                                                                              period_dslow=5)
                self.ind_dict[num+'_sto_daily_5'] = Stochastic_slow_daeshin(self.datas[i],
                                                                             period=5,
                                                                             period_dfast=3,
                                                                             period_dslow=3)
                self.ind_dict[num+'_MACD_daily'] = bt.ind.MACD(self.datas[i])

            elif i % 3 == 1:# weekly data
                self.ind_dict[num+'_sto_weekly_20'] = Stochastic_slow_daeshin(self.datas[i])
                self.ind_dict[num+'_sto_weekly_12'] = Stochastic_slow_daeshin(self.datas[i],
                                                                              period=12,
                                                                              period_dfast=5,
                                                                              period_dslow=5)
                self.ind_dict[num+'_sto_weekly_5'] = Stochastic_slow_daeshin(self.datas[i],
                                                                             period=5,
                                                                             period_dfast=3,
                                                                             period_dslow=3)
                self.ind_dict[num+'_MACD_weekly'] = bt.ind.MACD(self.datas[i])

            elif i % 3 == 2:# monthly data
                pass
                # self.ind_dict[num+'_sto_monthly_20'] = Stochastic_slow_daeshin(self.datas[i])
                # self.ind_dict[num+'_sto_monthly_5'] = Stochastic_slow_daeshin(self.datas[i],
                #                                                               period=5,
                #                                                               period_dfast=3,
                #                                                               period_dslow=3)
                # self.ind_dict[num+'_MACD_monthly'] = bt.ind.MACD(self.datas[i])

    def next(self):

        if self.holding.keys(): #보유하고 있는 주식이 있으면
            for i in self.holding.keys():
                self.holding[i] += 1 #보유 일수 증가

        # 최대 손익률 계산용 날짜 기록
        # self.record_date.append(self.datas[0].datetime.date(0).isoformat())
        self.record.append([])
        self.record[-1].append(self.datas[0].datetime.date(0).isoformat())
        self.record[-1].append(None)


        # if not self.position: # not in the market
        for i in range(len(self.datas)//3):
            # for i in range(0, len(self.datas)//3, 3):

            cond_A = self.ind_dict[str(i)+'_sto_daily_20'].percK[0] > self.ind_dict[str(i)+'_sto_daily_20'].percK[-1]
            cond_B = self.ind_dict[str(i)+'_sto_daily_20'].percK[0] >= self.ind_dict[str(i)+'_sto_daily_20'].percD[0]
            cond_C = self.ind_dict[str(i)+'_sto_daily_5'].percK[0] <= 25
            cond_D = self.ind_dict[str(i)+'_sto_weekly_20'].percD[0] > self.ind_dict[str(i)+'_sto_weekly_20'].percD[-1]
            cond_E = self.ind_dict[str(i)+'_sto_weekly_5'].percD[0] > self.ind_dict[str(i)+'_sto_weekly_5'].percD[-1]
            cond_F = self.ind_dict[str(i)+'_MACD_weekly'].macd[0] > self.ind_dict[str(i)+'_MACD_weekly'].macd[-1]
            cond_G = self.ind_dict[str(i)+'_sto_daily_12'].percK[0] <= 66
            cond_H = -900 <= self.ind_dict[str(i)+'_MACD_daily'].macd[0] <= 500
            cond_I = self.ind_dict[str(i)+'_sto_daily_20'].percK[0] <= 68 #TODO J조건이랑 번갈아가면서 비교
            # cond_J = self.ind_dict[str(i)+'_sto_daily_20'].percK[0] <= 36

            data_idx = i * 3
            if cond_A and cond_B and cond_C and cond_D and \
                    cond_E and cond_F and cond_G and cond_H and \
                    cond_I:
                print('종목 번호',i)
                close = self.datas[data_idx].close[0] # 종가 값
                size = int(self.broker.getcash() / close) # 최대 구매 가능 개수
                # size = int(size/2)z
                size = 1


                self.buy(size=size) # 매수 size = 구매 개수 설정

                self.log('BUY CREATE, %.2f' % self.datas[data_idx].close[0])

                # 보유기간 세기 위해
                self.holding[self.stock_num] = 0 #주식 구매 시 key 추가
                self.stock_num += 1

                # 보유기간 중 최대 손익률 검증용
                self.record[-1].append(self.datas[data_idx].close[0])

                if self.record[-1][1] == None:
                    self.record[-1][1] = str(self.name_list[i])
                else: #다른 종목도 시그널이 동일한 날짜에 발생했을 때
                    value = self.record[-1][1]
                    self.record[-1][1] = str(value)+'_'+str(self.name_list[i])


            else:#구매조건이 아닐 경우

                if self.date_count == 0:
                    self.record[0].append(None)

                #     self.record[i+1].append(None)
                elif self.record[-2][i+self.default_col] != None: # TODO 전날 구매했으면 종가 계속 기록

                    self.record[-1].append(self.datas[data_idx].close[0]) # TODO 왜 daily가 i-1이지
                elif False: # 매도 했으면 종가 기록 stop
                    pass
                else: # 보유도 안하고 구매도 안했으면 None 기록

                    self.record[-1].append(None)

        # else: # not in position
        if self.holding.keys():  #보유하고 있는 주식이 있으면
            key_list = copy.deepcopy(list(self.holding.keys()))
            for i in key_list:
                if self.holding[i] >= 60: #보유 일(봉)수
                    self.close() # 매도
                    # self.log('SELL CREATE, %.2f' % self.datas[i].close[0])
                    del self.holding[i]

        # 최대 손익률 계산용 결과 생성
        if self.record[-1][0] == '2021-07-13':
            result = pd.DataFrame(self.record, columns=self.column)
            # result.to_pickle('result/result_test.pickle')

        self.date_count += 1

# 매도 전략
