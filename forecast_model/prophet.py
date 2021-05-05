from statsmodels.tsa.stattools import adfuller

import os
from werkzeug.utils import secure_filename
import pandas as pd
from statsmodels.tools.eval_measures import rmse
from fbprophet import Prophet


class ProphetModel:
    accuracy=0
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



    def execute(self,filename):
        df1= pd.read_csv('data/'+filename,index_col='Date',parse_dates=True)
        df1.index.freq='MS'
        
        df = pd.read_csv('./data/'+filename)
        df.columns = ['ds','y']
        df['ds'] = pd.to_datetime(df['ds'])
        m = Prophet()
        m.fit(df)
        future = m.make_future_dataframe(periods=24,freq = 'MS')
        forecast = m.predict(future)
        filename=filename[:-4]
        m.plot(forecast).savefig(os.path.join('static/', secure_filename(filename+'_prophetPredict.jpg')))
        # 80% for training
        train = df.iloc[:int(len(df)*0.8)]
        test = df.iloc[len(train):]
        m = Prophet()
        m.fit(train)
        future = m.make_future_dataframe(periods=len(test),freq='MS')
        forecast = m.predict(future)
        #print(forecast.tail())
        ax = forecast.plot(x='ds',y='yhat',label='Predictions',legend=True,figsize=(12,8))
        g=test.plot(x='ds',y='y',label='True Miles',legend=True,ax=ax,xlim=('2018-01-01','2019-01-01'))
        g.figure.savefig(os.path.join('static/', secure_filename(filename+'_prophetCompare.jpg')))
        predictions = forecast.iloc[len(train):]['yhat']
        error=rmse(predictions,test['y'])
        mean=test.mean()
        print('percentage')
        self.accuracy=100-(error/mean*100)
        data=dict()
        data['stationary']=self.adf_test(df1)
        data['accuracy']=str(self.accuracy)
        return data



#a=ProphetModel()
#print(a.execute('BeerWineLiquor.csv'))