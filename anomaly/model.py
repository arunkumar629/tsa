from pycaret.anomaly import *
from werkzeug.utils import secure_filename
import pandas as pd
import shutil
import os
import plotly.express as px
import plotly.graph_objects as go
from pandas.api.types import is_datetime64_any_dtype as is_datetime

class anomaly:
    def abod(self,filename):
        df=pd.read_csv('data/'+filename)
        df[df.select_dtypes(['object']).columns]=df[df.select_dtypes(['object']).columns].apply(pd.to_datetime)
        for col in df.columns:
            if df[col].dtype == 'datetime64[ns]':
                df[col] = pd.to_datetime(df[col])
                df.set_index(col,drop=True, inplace=True)
        df['day'] = [i.day for i in df.index]
        df['day_name'] = [i.day_name() for i in df.index]
        df['day_of_year'] = [i.dayofyear for i in df.index]
        df['week_of_year'] = [i.weekofyear for i in df.index]
        df['is_weekday'] = [i.isoweekday() for i in df.index]
        exp_ano101 = setup(df, normalize = True , silent=True)
        model = create_model('abod')
        model_results = assign_model(model)
        # plot value on y-axis and date on x-axis
        fig = px.line(model_results, x=model_results.index, y="data", title='UNSUPERVISED ANOMALY DETECTION', template = 'plotly_dark')
        # create list of outlier_dates
        outlier_dates = model_results[model_results['Anomaly'] == 1].index
        # obtain y value of anomalies to plot
        y_values = [model_results.loc[i]['data'] for i in outlier_dates]
        fig.add_trace(go.Scatter(x=outlier_dates, y=y_values, mode = 'markers', 
                name = 'Anomaly', 
                marker=dict(color='red',size=10)))
        filename=filename[:-4]        
        fig.write_image(os.path.join('static/', secure_filename(filename+'_abod.jpg')))
        
    def cluster(self,filename):
        df=pd.read_csv('data/'+filename)
        df[df.select_dtypes(['object']).columns]=df[df.select_dtypes(['object']).columns].apply(pd.to_datetime)
        for col in df.columns:
            if df[col].dtype == 'datetime64[ns]':
                df[col] = pd.to_datetime(df[col])
                df.set_index(col,drop=True, inplace=True)
        df['day'] = [i.day for i in df.index]
        df['day_name'] = [i.day_name() for i in df.index]
        df['day_of_year'] = [i.dayofyear for i in df.index]
        df['week_of_year'] = [i.weekofyear for i in df.index]
        df['is_weekday'] = [i.isoweekday() for i in df.index]
        exp_ano101 = setup(df, normalize = True , silent=True)
        model = create_model('cluster')
        model_results = assign_model(model)
        # plot value on y-axis and date on x-axis
        fig = px.line(model_results, x=model_results.index, y="data", title='UNSUPERVISED ANOMALY DETECTION', template = 'plotly_dark')
        # create list of outlier_dates
        outlier_dates = model_results[model_results['Anomaly'] == 1].index
        # obtain y value of anomalies to plot
        y_values = [model_results.loc[i]['data'] for i in outlier_dates]
        fig.add_trace(go.Scatter(x=outlier_dates, y=y_values, mode = 'markers', 
                name = 'Anomaly', 
                marker=dict(color='red',size=10)))
        filename=filename[:-4]
        fig.write_image(os.path.join('static/', secure_filename(filename+'_cluster.jpg')))
        
    def cof(self,filename):
        df=pd.read_csv('data/'+filename)
        df[df.select_dtypes(['object']).columns]=df[df.select_dtypes(['object']).columns].apply(pd.to_datetime)
        for col in df.columns:
            if df[col].dtype == 'datetime64[ns]':
                df[col] = pd.to_datetime(df[col])
                df.set_index(col,drop=True, inplace=True)
        df['day'] = [i.day for i in df.index]
        df['day_name'] = [i.day_name() for i in df.index]
        df['day_of_year'] = [i.dayofyear for i in df.index]
        df['week_of_year'] = [i.weekofyear for i in df.index]
        df['is_weekday'] = [i.isoweekday() for i in df.index]
        exp_ano101 = setup(df, normalize = True , silent=True)
        model = create_model('cof')
        model_results = assign_model(model)
        # plot value on y-axis and date on x-axis
        fig = px.line(model_results, x=model_results.index, y="data", title='UNSUPERVISED ANOMALY DETECTION', template = 'plotly_dark')
        # create list of outlier_dates
        outlier_dates = model_results[model_results['Anomaly'] == 1].index
        # obtain y value of anomalies to plot
        y_values = [model_results.loc[i]['data'] for i in outlier_dates]
        fig.add_trace(go.Scatter(x=outlier_dates, y=y_values, mode = 'markers', 
                name = 'Anomaly', 
                marker=dict(color='red',size=10)))
        filename=filename[:-4]        
        fig.write_image(os.path.join('static/', secure_filename(filename+'_cof.jpg')))

    def iforest(self,filename):
        df=pd.read_csv('data/'+filename)
        df[df.select_dtypes(['object']).columns]=df[df.select_dtypes(['object']).columns].apply(pd.to_datetime)
        for col in df.columns:
            if df[col].dtype == 'datetime64[ns]':
                df[col] = pd.to_datetime(df[col])
                df.set_index(col,drop=True, inplace=True)
        df['day'] = [i.day for i in df.index]
        df['day_name'] = [i.day_name() for i in df.index]
        df['day_of_year'] = [i.dayofyear for i in df.index]
        df['week_of_year'] = [i.weekofyear for i in df.index]
        df['is_weekday'] = [i.isoweekday() for i in df.index]
        exp_ano101 = setup(df, normalize = True , silent=True)
        model = create_model('iforest',fraction = 0.1)
        model_results = assign_model(model)
        # plot value on y-axis and date on x-axis
        fig = px.line(model_results, x=model_results.index, y="data", title='UNSUPERVISED ANOMALY DETECTION', template = 'plotly_dark')
        # create list of outlier_dates
        outlier_dates = model_results[model_results['Anomaly'] == 1].index
        # obtain y value of anomalies to plot
        y_values = [model_results.loc[i]['data'] for i in outlier_dates]
        fig.add_trace(go.Scatter(x=outlier_dates, y=y_values, mode = 'markers', 
                name = 'Anomaly', 
                marker=dict(color='red',size=10)))
        filename=filename[:-4]
        fig.write_image(os.path.join('static/', secure_filename(filename+'_iforest.jpg')))
        
    def knn(self,filename):
        df=pd.read_csv('data/'+filename)
        df[df.select_dtypes(['object']).columns]=df[df.select_dtypes(['object']).columns].apply(pd.to_datetime)
        for col in df.columns:
            if df[col].dtype == 'datetime64[ns]':
                df[col] = pd.to_datetime(df[col])
                df.set_index(col,drop=True, inplace=True)
        df['day'] = [i.day for i in df.index]
        df['day_name'] = [i.day_name() for i in df.index]
        df['day_of_year'] = [i.dayofyear for i in df.index]
        df['week_of_year'] = [i.weekofyear for i in df.index]
        df['is_weekday'] = [i.isoweekday() for i in df.index]
        exp_ano101 = setup(df, normalize = True , silent=True)
        model = create_model('knn')
        model_results = assign_model(model)
        # plot value on y-axis and date on x-axis
        fig = px.line(model_results, x=model_results.index, y="data", title='UNSUPERVISED ANOMALY DETECTION', template = 'plotly_dark')
        # create list of outlier_dates
        outlier_dates = model_results[model_results['Anomaly'] == 1].index
        # obtain y value of anomalies to plot
        y_values = [model_results.loc[i]['data'] for i in outlier_dates]
        fig.add_trace(go.Scatter(x=outlier_dates, y=y_values, mode = 'markers', 
                name = 'Anomaly', 
                marker=dict(color='red',size=10)))
        filename=filename[:-4]
        fig.write_image(os.path.join('static/', secure_filename(filename+'_knn.jpg')))
                                        
    def lof(self,filename):
        df=pd.read_csv('data/'+filename)
        df[df.select_dtypes(['object']).columns]=df[df.select_dtypes(['object']).columns].apply(pd.to_datetime)
        for col in df.columns:
            if df[col].dtype == 'datetime64[ns]':
                df[col] = pd.to_datetime(df[col])
                df.set_index(col,drop=True, inplace=True)
        df['day'] = [i.day for i in df.index]
        df['day_name'] = [i.day_name() for i in df.index]
        df['day_of_year'] = [i.dayofyear for i in df.index]
        df['week_of_year'] = [i.weekofyear for i in df.index]
        df['is_weekday'] = [i.isoweekday() for i in df.index]
        exp_ano101 = setup(df, normalize = True , silent=True)
        model = create_model('lof')
        model_results = assign_model(model)
        # plot value on y-axis and date on x-axis
        fig = px.line(model_results, x=model_results.index, y="data", title='UNSUPERVISED ANOMALY DETECTION', template = 'plotly_dark')
        # create list of outlier_dates
        outlier_dates = model_results[model_results['Anomaly'] == 1].index
        # obtain y value of anomalies to plot
        y_values = [model_results.loc[i]['data'] for i in outlier_dates]
        fig.add_trace(go.Scatter(x=outlier_dates, y=y_values, mode = 'markers', 
                name = 'Anomaly', 
                marker=dict(color='red',size=10)))
        filename=filename[:-4]        
        fig.write_image(os.path.join('static/', secure_filename(filename+'_loc.jpg')))

    def svm(self,filename):
        df=pd.read_csv('data/'+filename)
        df[df.select_dtypes(['object']).columns]=df[df.select_dtypes(['object']).columns].apply(pd.to_datetime)
        for col in df.columns:
            if df[col].dtype == 'datetime64[ns]':
                df[col] = pd.to_datetime(df[col])
                df.set_index(col,drop=True, inplace=True)
        df['day'] = [i.day for i in df.index]
        df['day_name'] = [i.day_name() for i in df.index]
        df['day_of_year'] = [i.dayofyear for i in df.index]
        df['week_of_year'] = [i.weekofyear for i in df.index]
        df['is_weekday'] = [i.isoweekday() for i in df.index]
        exp_ano101 = setup(df, normalize = True , silent=True)
        model = create_model('svm')
        model_results = assign_model(model)
        # plot value on y-axis and date on x-axis
        fig = px.line(model_results, x=model_results.index, y="data", title='UNSUPERVISED ANOMALY DETECTION', template = 'plotly_dark')
        # create list of outlier_dates
        outlier_dates = model_results[model_results['Anomaly'] == 1].index
        # obtain y value of anomalies to plot
        y_values = [model_results.loc[i]['data'] for i in outlier_dates]
        fig.add_trace(go.Scatter(x=outlier_dates, y=y_values, mode = 'markers', 
                name = 'Anomaly', 
                marker=dict(color='red',size=10)))
        filename=filename[:-4]
        fig.write_image(os.path.join('static/', secure_filename(filename+'_svm.jpg')))
        
    def sod(self,filename):
        df=pd.read_csv('data/'+filename)
        df[df.select_dtypes(['object']).columns]=df[df.select_dtypes(['object']).columns].apply(pd.to_datetime)
        for col in df.columns:
            if df[col].dtype == 'datetime64[ns]':
                df[col] = pd.to_datetime(df[col])
                df.set_index(col,drop=True, inplace=True)
        df['day'] = [i.day for i in df.index]
        df['day_name'] = [i.day_name() for i in df.index]
        df['day_of_year'] = [i.dayofyear for i in df.index]
        df['week_of_year'] = [i.weekofyear for i in df.index]
        df['is_weekday'] = [i.isoweekday() for i in df.index]
        exp_ano101 = setup(df, normalize = True , silent=True)
        model = create_model('sod')
        model_results = assign_model(model)
        # plot value on y-axis and date on x-axis
        fig = px.line(model_results, x=model_results.index, y="data", title='UNSUPERVISED ANOMALY DETECTION', template = 'plotly_dark')
        # create list of outlier_dates
        outlier_dates = model_results[model_results['Anomaly'] == 1].index
        # obtain y value of anomalies to plot
        y_values = [model_results.loc[i]['data'] for i in outlier_dates]
        fig.add_trace(go.Scatter(x=outlier_dates, y=y_values, mode = 'markers', 
                name = 'Anomaly', 
                marker=dict(color='red',size=10)))
        filename=filename[:-4]
        fig.write_image(os.path.join('static/', secure_filename(filename+'_sod.jpg')))
                                        
    def histogram(self,filename):
        df=pd.read_csv('data/'+filename)
        df[df.select_dtypes(['object']).columns]=df[df.select_dtypes(['object']).columns].apply(pd.to_datetime)
        for col in df.columns:
            if df[col].dtype == 'datetime64[ns]':
                df[col] = pd.to_datetime(df[col])
                df.set_index(col,drop=True, inplace=True)
        df['day'] = [i.day for i in df.index]
        df['day_name'] = [i.day_name() for i in df.index]
        df['day_of_year'] = [i.dayofyear for i in df.index]
        df['week_of_year'] = [i.weekofyear for i in df.index]
        df['is_weekday'] = [i.isoweekday() for i in df.index]
        exp_ano101 = setup(df, normalize = True , silent=True)
        model = create_model('histogram')
        model_results = assign_model(model)
        # plot value on y-axis and date on x-axis
        fig = px.line(model_results, x=model_results.index, y="data", title='UNSUPERVISED ANOMALY DETECTION', template = 'plotly_dark')
        # create list of outlier_dates
        outlier_dates = model_results[model_results['Anomaly'] == 1].index
        # obtain y value of anomalies to plot
        y_values = [model_results.loc[i]['data'] for i in outlier_dates]
        fig.add_trace(go.Scatter(x=outlier_dates, y=y_values, mode = 'markers', 
                name = 'Anomaly', 
                marker=dict(color='red',size=10)))
        filename=filename[:-4]
        fig.write_image(os.path.join('static/', secure_filename(filename+'_histogram.jpg')))
                                                                          
                                                       
#ano=anomaly()
#ano.abod('HospitalityEmployees.csv')
#ano.cluster('HospitalityEmployees.csv')
#ano.cof('HospitalityEmployees.csv')
#ano.iforest('HospitalityEmployees.csv')
#ano.knn('HospitalityEmployees.csv')
#ano.lof('HospitalityEmployees.csv')
#ano.svm('HospitalityEmployees.csv             ')
#ano.sod('HospitalityEmployees.csv')
#ano.histogram('HospitalityEmployees.csv')