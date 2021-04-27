from pycaret.anomaly import *
import pandas as pd
class anomaly:
    def create(self,filename):
        dataset=pd.read_csv('data/'+filename, usecols=['data'])
        data = dataset.sample(frac=0.95, random_state=786)
        data_unseen = dataset.drop(data.index)
        data.reset_index(drop=True, inplace=True)
        data_unseen.reset_index(drop=True, inplace=True)
        exp_ano101 = setup(data, normalize = True )
        iforest = create_model('iforest')
        svm = create_model('svm', fraction = 0.025)
        iforest_results = assign_model(iforest)
        plot_model(iforest, save=True)
