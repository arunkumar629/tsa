import numpy as np 
import pandas as pd 
import os
from werkzeug.utils import secure_filename
class Visualization:
    def chart(self,filename):
        df = pd.read_csv('data/'+filename,index_col='Month',parse_dates=True)
        df.to_csv(os.path.join('static/', secure_filename(filename)))
