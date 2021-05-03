import os
from werkzeug.utils import secure_filename
import pandas as pd
from fbprophet import Prophet


class ProphetModel:
    def execute(self,filename):
        df = pd.read_csv('./data/'+filename)
        df.columns = ['ds','y']
        df['ds'] = pd.to_datetime(df['ds'])
        m = Prophet()
        m.fit(df)
        future = m.make_future_dataframe(periods=24,freq = 'MS')
        forecast = m.predict(future)
        filename=filename[:-4]
        m.plot(forecast).savefig(os.path.join('static/', secure_filename(filename+'_prophetPredict.jpg')))


#a=ProphetModel()
#a.execute('BeerWineLiquor.csv')