import tensorflow as tf
import data
import time
import pandas as pd
import pickle
import numpy as np
import lstm
from numpy import newaxis
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import keras
import matplotlib.pyplot as plt
import math
import time

from peakdetect import peakdetect #cite https://gist.github.com/sixtenbe/1178136

#data collection and it will run in data.py and saved as pickle file format for next usage
#closeddf = data.collect("Close")
#highdf = data.collect("High")
#lowdf = data.collect("Low")

#pre-saved data in pickle, open it for quicker run.
pickle_in = open("Close.pickle","rb")
closeddf = pickle.load(pickle_in)
pickle_in = open("Low.pickle","rb")
lowdf = pickle.load(pickle_in)
pickle_in = open("High.pickle","rb")
highdf = pickle.load(pickle_in)
#different indicators is calculated by calling back to data.py and save it as the particular
#pandas database
#normalizing the closeddf by using the first day closing price
normalizedCloseddf = data.normalizeData(closeddf)
volumeDF = data.readdata("Volume")
SMAdf = data.SMA(closeddf,20)
EMAdf = data.EMA(closeddf,5)
RSIdf = data.RSI(closeddf,14)
KSTdf = data.KST(closeddf,10,15,20,30,10,10,10,15)
TRIXdf = data.TRIX(closeddf,15)
IXIC = pd.DataFrame(index=normalizedCloseddf.index)
IXIC = normalizedCloseddf[['IXIC Close']]
highbbdf, lowbbdf = data.BollingerBand(closeddf)
MassIdf = data.MassI(highdf,lowdf)
MOMdf = data.MOM(closeddf,10)
ROCdf = data.ROC(closeddf,10)
dailyReturnDF = data.dailyReturn(normalizedCloseddf)
MACDdf,MACDsigndf, MACDDiffdf = data.MACD(closeddf)
STOKdf = data.STOK(closeddf,lowdf,highdf)
STOdf = data.STO(closeddf,lowdf,highdf,14)


#SETTINGS for Stock, epoch cycles and time frame
#'GOOGL','MSFT','AAPL', 'AMZN','FB'
tickers = 'GOOGL','MSFT','AAPL', 'AMZN'
epoch=500
timeframe = 15

#Indicators combination is below
indicators = [TRIXdf,STOKdf,MOMdf,normalizedCloseddf]
indicators2 = [STOdf,MOMdf,MACDDiffdf,normalizedCloseddf]
indicators3 = [highbbdf, lowbbdf,RSIdf,MACDDiffdf,normalizedCloseddf]
indicators4 = [highbbdf, lowbbdf,ROCdf,MACDDiffdf,normalizedCloseddf]
indicators5 = [highbbdf, lowbbdf,TRIXdf,MACDDiffdf,normalizedCloseddf]
indicators6 = [STOKdf,KSTdf,ROCdf,MACDDiffdf,normalizedCloseddf]
indicators7 = [MassIdf,MOMdf,normalizedCloseddf]
indicators8 = [ROCdf, TRIXdf,MOMdf, normalizedCloseddf]
indicators9 = [ROCdf,STOKdf,MOMdf,MACDDiffdf,normalizedCloseddf]
indicators10= [TRIXdf,MassIdf,MOMdf,normalizedCloseddf]
indicators11= [highbbdf, lowbbdf, MOMdf,normalizedCloseddf]
indicators12= [STOKdf,MOMdf,MACDDiffdf,normalizedCloseddf]
indicators13= [TRIXdf,MOMdf,normalizedCloseddf]
indicators14= [TRIXdf,MOMdf,MACDDiffdf,normalizedCloseddf]
indicators15= [TRIXdf,STOKdf,MOMdf,normalizedCloseddf]
indicators16= [volumeDF, MACDDiffdf,normalizedCloseddf]
indicators17= [SMAdf,KSTdf,RSIdf,normalizedCloseddf]
indicators18= [EMAdf,MassIdf,normalizedCloseddf]
indicators19= [SMAdf,RSIdf,normalizedCloseddf]
indicators20= [SMAdf,MACDDiffdf, KSTdf,normalizedCloseddf]
indicators21= [IXIC, STOKdf,KSTdf,ROCdf,MACDDiffdf,normalizedCloseddf]
totalindictors = [indicators12]

def main(ticker, epoch,timeframe, indictors):
    stock = pd.DataFrame(index=normalizedCloseddf.index)
    #Extract the ticker
    #Stick all indicators in the combination which fits the company to dataframe "stock"
    for indictor in indictors:
            if str(indictor) == str(IXIC):
                stock = stock.join(IXIC)
            stock = stock.join(indictor.filter(regex=ticker))
    #startDate is marked the date, user wants to start the dataset.
    startDate = stock.index.get_loc('2005-01-01')
    stock = stock.drop(stock.index[0:startDate])
    print(stock.head(5))

    #this function will load the first 20% of data as testing and 80% till the latest as training data
    X_train, y_train, X_test, y_test = lstm.load_dataforward(stock[::-1], timeframe)
    #this function will load the last 20% of data as testing and 80% from the beginning as training data
    #X_train, y_train, X_test, y_test = lstm.load_data(stock[::-1], 30)
    print("X_train", X_train.shape)
    print("y_train", y_train.shape)
    print("X_test", X_test.shape)
    print("y_test", y_test.shape)
    #tbcallback will generate a graph which can show back in tensorboard
    tbCallBack = keras.callbacks.TensorBoard(log_dir="./Graph", histogram_freq=0, write_graph=True, write_images=True)
    #model will have three variables, the first variable is number of indicators, the second variable is the timeframe
    model = lstm.model([len(indictors),timeframe])
    model.fit(
        X_train,
        y_train,
        batch_size=150,
        nb_epoch=epoch,
        validation_split=0.2,
        callbacks=[tbCallBack],
        verbose=1)
    model.save(ticker+'.h5')

    trainScore = model.evaluate(X_train, y_train, verbose=0)
    print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore[0], math.sqrt(trainScore[0])))

    testScore = model.evaluate(X_test, y_test, verbose=0)
    print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore[0], math.sqrt(testScore[0])))
    prediction = model.predict(X_test)
    prediction = prediction[::-1]
    y_test = y_test[::-1]
    peakperiod=5
    peaksdetect = peakdetect(prediction, lookahead=peakperiod)
    peaksdetect = np.array(peaksdetect)
    peaks = peaksdetect[0]
    toughs = peaksdetect[1]
    peakpoints = [item[1] for item in peaks]
    peakdays = [item[0] for item in peaks]
    toughpoints = [item[1] for item in toughs]
    toughdays = [item[0] for item in toughs]
    plt.figure(figsize=(18,7))
    plt.plot(prediction,color='red', label='prediction')
    plt.scatter(peakdays, peakpoints, color='red',marker='x')
    plt.scatter(toughdays, toughpoints, color='green',marker='o')
    plt.plot(y_test,color='blue', label='y_test')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.subplots_adjust(top=0.95,left=0.08,bottom=0.08)
    strIndictors = ''.join(str(e) for e in indictors)
    title=epoch," epoch ",timeframe, ' timeframe',ticker,list(stock)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Price")
    listIndicators = list(stock)
    epochstr = str(epoch)
    timeframestr = str(timeframe)
    peakperiod = str(peakperiod)
    strIndicators = ''.join(listIndicators)
    csvname = epochstr +' e'+ timeframestr +' t'+ strIndicators + ' peak '+peakperiod+'.csv'
    np.savetxt(csvname,np.c_[prediction,y_test],delimiter=",")
    filename = epochstr +' e'+ timeframestr +' t'+ strIndicators + ' peak '+peakperiod+'.png'
    plt.savefig(filename)


for indictor in totalindictors:
    for ticker in tickers:
        start = time.time()
        main(ticker, epoch,timeframe, indictor)
        end = time.time()
        runTime= int(end-start)
        print('Run time: ', runTime,' seconds')
