import os
from werkzeug.utils import secure_filename
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller

# Load specific forecasting tools
from statsmodels.tsa.arima_model import ARMA,ARMAResults,ARIMA,ARIMAResults
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf # for determining (p,q) orders
from pmdarima import auto_arima # for determining ARIMA orders

# Ignore harmless warnings
import warnings
warnings.filterwarnings("ignore")


class ArimaModel:
    accuracy=0
    def timeRange(self,filename):
        df= pd.read_csv('data/'+filename,index_col='Date',parse_dates=True)
        df.index.freq='MS'
        length=len(df)-1
        start_month= str(df.index[0])
        end_month= str(df.index[length])
        drange= pd.date_range(end_month, periods=11, freq='MS')
        forecasted_length=str(drange[10])
        return {"start":start_month,"end":end_month,"forecasted_month":forecasted_length,"accuracy":self.accuracy}

        
    def create_model(self,filename):
        df2 = pd.read_csv('data/'+filename,index_col='Date',parse_dates=True)
        filename=filename[:-4]
        df2.index.freq='MS'
        stepwise_fit = auto_arima(df2['data'], start_p=0, start_q=0,
                          max_p=2, max_q=2, m=12,
                          seasonal=False,
                          d=None, trace=True,
                          error_action='ignore',   # we don't want to know if an order does not work
                          suppress_warnings=True,  # we don't want convergence warnings
                          stepwise=True)           # set to stepwise
        # 80% for training
        train = df2.iloc[:int(len(df2)*0.8)]
        test = df2.iloc[int(len(df2)*0.2):]
        model = ARIMA(train['data'],order=(1,1,1))
        results = model.fit()
        start=len(train)
        end=len(train)+len(test)-1
        predictions = results.predict(start=start, end=end, dynamic=False, typ='levels').rename('ARIMA(1,1,1) Predictions')
        title = 'Real Manufacturing and Trade Inventories'
        ylabel='Chained 2012 Dollars'
        xlabel='' # we don't really need a label here
        #ax = test['data'].plot(legend=True,figsize=(12,6),title=title)
        #predictions.plot(legend=True)
        #ax.autoscale(axis='x',tight=True)
        #ax.set(xlabel=xlabel, ylabel=ylabel)
        #ax.yaxis.set_major_formatter(formatter)
        #ax.figure.savefig('fig.jpg')
        error = mean_squared_error(test['data'], predictions)
        #print(f'ARIMA(1,1,1) MSE Error: {error:11.10}')
        model = ARIMA(df2['data'],order=(1,1,1))
        results = model.fit()
        fcast = results.predict(len(df2),len(df2)+11,typ='levels').rename('ARIMA(1,1,1) Forecast')
        # Plot predictions against known values
        title = filename
        ylabel=''
        xlabel='' # we don't really need a label here'
        ax = df2['data'].plot(legend=True,figsize=(12,6),title=title)
        fcast.plot(legend=True)
        ax.autoscale(axis='x',tight=True)
        ax.set(xlabel=xlabel, ylabel=ylabel)
        #ax.yaxis.set_major_formatter(formatter);
        ax.figure.savefig(os.path.join('static/', secure_filename(filename+'_forecast.jpg')))
        print('mse error:',error)
        mean=df2['data'].mean()
        rmse=np.sqrt(error)
        error_percentage=rmse/mean*100
        self.accuracy=100-error_percentage
        


    def adf_test(self,series,title=''):
        """
        Pass in a time series and an optional title, returns an ADF report
        """
        #print(f'Augmented Dickey-Fuller Test: {title}')
        result = adfuller(series.dropna(),autolag='AIC') # .dropna() handles differenced data
        labels = ['ADF test statistic','p-value','# lags used','# observations']
        out = pd.Series(result[0:4],index=labels)

        for key,val in result[4].items():
            out[f'critical value ({key})']=val
        
            #print(out.to_string())          # .to_string() removes the line "dtype: float64"
    
        if result[1] <= 0.05:
            return True
        else:
            return False


#ar=ArimaModel()
#ar.create_model('TradeInventories.csv')