from math import nan
import pandas as pd

from wandb import save

df_name="/home/jixy/naru/utils/distill/tables/alpha0.85_result_card.csv"


cols=["Dataset","Model","Type","Train_len","Mean","Median","queries","alpha",]

df=pd.read_csv(df_name,usecols=cols)

df=df[df["queries"]==200]
df=df[df["Type"]=="distill"]
# df["mixup"]=df["mixup"]==True
df['$\\alpha$']=[a if not pd.isnull(a) else 0.8 for a in df['alpha']]


groups_tr = df.groupby("Train_len")

save_cols=["Dataset","Model","Mean","Median",'$\\alpha$']
subset=["Dataset","Model","alpha"]

import re
def process_set(set,value):
    set_new=[]
    for name in set:
        if name=="DMV_ofnan":
            set_new.append("$DMV$")
        else:
            set_new.append("$DMV+D_i^{"+"{}".format(int(value))+"}$")
    return set_new

for value_tr, group_tr in groups_tr:
    
    
        
    group_tr=group_tr.sort_values(by=["Dataset","Model","Median",'$\\alpha$'])

    group_tr.drop_duplicates(subset=subset,keep='first',inplace=True)
    group_tr['Dataset']=process_set(group_tr['Dataset'],value_tr)
    group_tr.to_csv(f"distill/table_report/{value_tr}_report.csv",columns=save_cols)
        