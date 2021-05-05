from pycaret.anomaly import *
import pandas as pd
import shutil
import os

class anomaly:
    def create(self,filename):
        dataset=pd.read_csv('data/'+filename, usecols=['data'])
        data = dataset.sample(frac=0.95, random_state=786)
        data_unseen = dataset.drop(data.index)
        data.reset_index(drop=True, inplace=True)
        data_unseen.reset_index(drop=True, inplace=True)
        exp_ano101 = setup(data, normalize = True , silent=True)
        iforest = create_model('iforest')
        svm = create_model('svm', fraction = 0.025)
        iforest_results = assign_model(iforest)
        plot_model(iforest, save=True)
        l=os.listdir()
        for file in l:
            if 'html' ==file[-4:]:
                image_name=file

        try:
            shutil.move(image_name,os.path.join('static/','anomaly.html' ))
        except:
            print('no file ')



#ano=anomaly()
#ano.create('TradeInventories.csv')