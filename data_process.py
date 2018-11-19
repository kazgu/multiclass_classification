
import pandas as pd
data=pd.read_csv('testb_predict.csv')
data['content']=''
data.to_csv('testb_pre.csv',encoding="utf_8_sig", index=False)
