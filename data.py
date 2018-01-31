import pandas as pd
import numpy as np
import pandas_datareader.data as web
import datetime
import pickle
from pandas import DataFrame

#this will load the data from IXIC and the stocks tickers and store it in a dataframe
#it can be either in volume or in close price

def readdata(data):
    df = pd.read_csv("IXIC.csv", index_col="Date",parse_dates= True,na_values=["nan"])
    df = df.rename(columns={data : "IXIC "+ data})
    tickers = ["AAPL","GOOGL","MSFT","FB","AMZN","NFLX","C","WFC"]
    start = datetime.datetime(1980, 1, 1)
    end = datetime.datetime(2016, 12, 31)
    dates=pd.date_range(start,end)
    closedDataFrame = pd.DataFrame(index=dates)
    closedDataFrame["IXIC " + data] = df["IXIC " + data]
    closedDataFrame.fillna(method="ffill",inplace=True)
    closedDataFrame.fillna(method="bfill",inplace=True)
    closedDataFrame = closedDataFrame.dropna()
    for ticker in tickers:
        database = web.DataReader(ticker, 'google', start, end)
        closedDataFrame[ticker + " " + data] = database[data]
        closedDataFrame.fillna(method="ffill",inplace=True)
        closedDataFrame.fillna(method="bfill",inplace=True)
        closedDataFrame.astype(float)
    return closedDataFrame

#Mass Index
def MassI(df,df2):
    df = df.rename(columns=lambda x: x.replace('High', ''))
    df2 = df2.rename(columns=lambda x: x.replace('Low', ''))
    Range= pd.DataFrame(index=df.index)
    for ticker in df:
        Range[ticker + " MassI"] = df[ticker] - df2[ticker]
    EX1 = pd.ewma(Range, span = 9, min_periods = 8)
    EX2 = pd.ewma(EX1, span = 9, min_periods = 8)
    Mass = EX1 / EX2
    MassI = pd.rolling_sum(Mass, 25)
    return MassI

#Trix
def TRIX(df, n):
    EX1 = pd.ewma(df, span = n, min_periods = n - 1)
    EX2 = pd.ewma(EX1, span = n, min_periods = n - 1)
    EX3 = pd.ewma(EX2, span = n, min_periods = n - 1)
    TRIX = EX3.pct_change()
    TRIX = TRIX.rename(columns=lambda x: x.replace('Close', 'TRIX'))
    return TRIX

def dailyReturn(df):
    dailyReturn = pd.DataFrame(index=df.index)
    dailyReturn[1:]=(df[1:]/df[:-1].values)-1
    dailyReturn.ix[0,:]=0
    dailyreturn = dailyReturn.rename(columns=lambda x: x.replace('Close', 'Daily Return'))
    return dailyReturn

#stock price must be normalized or else no stock prediction can be made
def normalizeData(df):
    return df/ df.ix[0,:]


def SMA(df,window):
    sma = df.rolling(window=5, center=False).mean()
    sma = sma.rename(columns=lambda x: x.replace('Close', 'SMA'))
    return sma

#Stochastic oscillator %K
def STOK(closeddf,lowdf,highdf):
    SOk = pd.DataFrame(index=closeddf.index)
    closeddf = closeddf.rename(columns=lambda x: x.replace('Close', ''))
    lowdf = lowdf.rename(columns=lambda x: x.replace('Low', ''))
    highdf = highdf.rename(columns=lambda x: x.replace('High', ''))
    for column in closeddf:
        SOkcal = ((closeddf[column] - lowdf[column]) / (highdf[column] - lowdf[column])).to_frame(column + 'STOK')
        SOk = SOk.join(SOkcal)
    return SOk

#Stochastic oscillator %D
def STO(closeddf,lowdf,highdf, n):
    closeddf = closeddf.rename(columns=lambda x: x.replace('Close', ''))
    lowdf = lowdf.rename(columns=lambda x: x.replace('Low', ''))
    highdf = highdf.rename(columns=lambda x: x.replace('High', ''))
    SOk = pd.DataFrame(index=closeddf.index)
    SOd = pd.DataFrame(index=closeddf.index)
    for column in closeddf:
        SOkcal = ((closeddf[column] - lowdf[column]) / (highdf[column] - lowdf[column])).to_frame(column + 'STO')
        SOk = SOk.join(SOkcal)
    SOd = pd.ewma(SOk, span = n, min_periods = n - 1)
    return SOd

def EMA(df, n):
    EMA = pd.ewma(df, span = n, min_periods = n - 1)
    EMA = EMA.rename(columns=lambda x: x.replace('Close', 'EMA'))
    return EMA

def MOM(df, n):
    M = df.diff(n)
    M = M.rename(columns=lambda x: x.replace('Close', 'MOM'))
    return M
def MACD(df):
    slowMACDdf = pd.DataFrame(index=df.index)
    fastMACDdf = pd.DataFrame(index=df.index)
    MACDdf = pd.DataFrame(index=df.index)
    totalMACD = pd.DataFrame(index=df.index)
    slowMACDdf = EMA(df, 12)
    fastMACDdf = EMA(df, 26)
    MACDdf = fastMACDdf - slowMACDdf
    MACDdf = MACDdf.rename(columns=lambda x: x.replace('EMA','MACD'))
    MACDsign = pd.ewma(MACDdf, span = 9, min_periods = 8)
    MACDdiff = MACDdf-MACDsign
    MACDdiff = MACDdiff.rename(columns=lambda x: x.replace('MACD','MACDdiff'))
    MACDsign = MACDsign.rename(columns=lambda x: x.replace('MACD','MACDSign'))
    return MACDdf, MACDsign, MACDdiff

#Relative Strength Index
def RSI(df, period=14):
    # wilder's RSI
    delta = df.diff()
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    rUp = up.ewm(com=period - 1,  adjust=False).mean()
    rDown = down.ewm(com=period - 1, adjust=False).mean().abs()
    rsi = 100 - 100 / (1 + rUp / rDown)
    rsi = rsi.rename(columns=lambda x: x.replace('Close', 'RSI'))
    return rsi

#Bollinger Bands
def BollingerBand(df, period=20):
    highbbdf= pd.DataFrame(index=df.index)
    lowbbdf= pd.DataFrame(index=df.index)
    df = df.rename(columns=lambda x: x.replace('Close', ''))
    for column in df:
        sma = df[column].rolling(window=period, min_periods=period - 1).mean()
        std = df[column].rolling(window=period, min_periods=period - 1).std()
        up = (sma + (std * 2)).to_frame(column + 'BBANDUP')
        down = (sma - (std * 2)).to_frame(column + 'BBANDLO')
        highbbdf = highbbdf.join(up)
        lowbbdf = lowbbdf.join(down)
    return highbbdf, lowbbdf

#Rate of Change
def ROC(df, n):
    M = df.diff(n - 1)
    N = df.shift(n - 1)
    ROC = M / N
    ROC = ROC.rename(columns=lambda x: x.replace('Close', 'ROC'))
    return ROC

#KST Oscillator
def KST(df, r1, r2, r3, r4, n1, n2, n3, n4):
    M = df.diff(r1 - 1)
    N = df.shift(r1 - 1)
    ROC1 = M / N
    M = df.diff(r2 - 1)
    N = df.shift(r2 - 1)
    ROC2 = M / N
    M = df.diff(r3 - 1)
    N = df.shift(r3 - 1)
    ROC3 = M / N
    M = df.diff(r4 - 1)
    N = df.shift(r4 - 1)
    ROC4 = M / N
    KST = pd.rolling_sum(ROC1, n1) + pd.rolling_sum(ROC2, n2) * 2 + pd.rolling_sum(ROC3, n3) * 3 + pd.rolling_sum(ROC4, n4) * 4
    KST = KST.rename(columns=lambda x: x.replace('Close', 'KST'))
    return KST

def collect(data):
    df = readdata(data)
    pickle_out = open(data + ".pickle","wb")
    pickle.dump(df, pickle_out)
    pickle_out.close()
    return df
