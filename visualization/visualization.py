from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 
import os
from werkzeug.utils import secure_filename
class Visualization:
    def chart(self,filename):
        df = pd.read_csv('data/'+filename,index_col=0,parse_dates=True)
        #df.to_csv(os.path.join('static/', secure_filename(filename)))     
        df.dropna(inplace=True)

        #basic graph
        basic=df.plot()
        filename=filename[:-4]
        basic.figure.savefig(os.path.join('static/', secure_filename(filename+'_basic.jpg')))
        #decompose
        result = seasonal_decompose(df['data'], model='multiplicative')  
        ds=result.plot()
        ds.savefig(os.path.join('static/', secure_filename(filename+'_decompose.jpg')))

        #Simple moving average
        df['6-month-SMA']  = df.iloc[:,0].rolling(window=6).mean()
        df['12-month-SMA'] = df.iloc[:,0].rolling(window=12).mean()
        sma=df.plot()
        sma.figure.savefig(os.path.join('static/', secure_filename(filename+'_sma.jpg')))

        #Weighted moving average
        df['EWMA12'] = df.iloc[:,0].ewm(span=12,adjust=False).mean()
        wma=df.plot()
        wma.figure.savefig(os.path.join('static/', secure_filename(filename+'_wma.jpg')))

        #Exponential weighted moving average
        #df.index.freq = 'MS'
        span=12
        alpha = 2/(span+1)
        df['EWMA12'] = df.iloc[:,0].ewm(alpha=alpha,adjust=False).mean()
        df['SES12']=SimpleExpSmoothing(df.iloc[:,0]).fit(smoothing_level=alpha,optimized=False).fittedvalues.shift(-1)
        df['DESadd12'] = ExponentialSmoothing(df.iloc[:,0], trend='add').fit().fittedvalues.shift(-1)
        ewma=df[['data','EWMA12','DESadd12']].iloc[:24].plot()
        #ewma.autoscale(axis='x',tight=True)
        ewma.figure.savefig(os.path.join('static/', secure_filename(filename+'_ewma.jpg')))


