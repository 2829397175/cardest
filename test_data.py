import numpy as np
import pandas as pd

df=pd.read_csv("/home/jixy/naru/datasets/dmv-tiny.csv")
df_1=df.iloc[:10,:]
df_2=df.iloc[:11,:]

df_new=pd.concat([df_1,df_2])
df_new.iloc[0,0]=0

assert df_new.iloc[0,0]!=df.iloc[0,0]
assert df_new.shape[0]==21 and df_new.shape[1]==df.shape[1]