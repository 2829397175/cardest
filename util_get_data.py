import pandas as pd

# csv= pd.read_csv("/home/jixy/naru/df_dmvsets/dmv-tiny.csv")

# csv = csv.iloc[:10]
# csv.to_csv("/home/jixy/naru/df_dmvsets/insert_dmv.csv")

# import torch
# from torch.nn import functional as F
# a=torch.rand(3,16,20)
 
# b=F.softmax(a,dim=0)
 
# c=F.softmax(a,dim=1)
 
# d=F.softmax(a,dim=2)

# print(F.cross_entropy(d,d))
import numpy as np
df_dmv=pd.read_csv("/home/jixy/naru/datasets/Vehicle__Snowmobile__and_Boat_Registrations_modified.csv")
print(df_dmv['Reg Valid Date'][:5])

type_casts = {'Reg Valid Date': np.datetime64}
for col, typ in type_casts.items():
    if col not in df_dmv:
        continue
    if typ != np.datetime64:
        df_dmv[col] = df_dmv[col].astype(typ, copy=False)
    else:
        # Both infer_datetime_format and cache are critical for perf.
        df_dmv[col] = pd.to_datetime(df_dmv[col],
                                    infer_datetime_format=True,
                                    cache=True)

cols = [
        'Record Type', 'Registration Class', 'State', 'County', 'Body Type',
        'Fuel Type', 'Reg Valid Date', 'Color', 'Scofflaw Indicator',
        'Suspension Indicator', 'Revocation Indicator'
    ]

df_dmv = df_dmv[cols]
       
def judge_data(data):

    if isinstance(data['Reg Valid Date'],pd.Timestamp):
        return data['Reg Valid Date']
    else:
        return pd.Timestamp('2000-01-01')


df_dmv['Reg Valid Date']=df_dmv.apply(lambda x: judge_data(x), axis=1)
# df_dmv=df_dmv[df_dmv['Reg Valid Date']!=False]
df_dmv.to_csv("/home/jixy/naru/datasets/Vehicle__Snowmobile__and_Boat_Registrations_modified.csv")
