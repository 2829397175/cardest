import pandas as pd

cols = [
    'Record Type', 'Registration Class', 'State', 'County', 'Body Type',
    'Fuel Type', 'Reg Valid Date', 'Color', 'Scofflaw Indicator',
    'Suspension Indicator', 'Revocation Indicator'
]
# df=pd.read_csv("/home/jixy/naru/datasets/dmv_ofnan.csv",usecols=cols)

# shape_p2=int(df.shape[0]*0.1)
df_insert=pd.read_csv("/home/jixy/naru/datasets/insert_dmv_200000.csv",usecols=cols)
df_insert=df_insert[:20000]
df_insert.to_csv("/home/jixy/naru/datasets/insert_dmv_20000.csv")


# df=df[:shape_60]
# df.to_csv("/home/jixy/naru/datasets/dmv_p60.csv")

# cols=[473, 5, 9, 10, 11, 12, 52, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 74, 6, 17, 18, 361, 19, 20, 21, 22, 42, 73, 470, 472, 475, 477, 50, 476, 478, 318, 25, 51, 54, 408, 474, 358, 384, 14, 317, 326, 350, 161, 234, 148, 324, 13, 147, 88, 299, 316, 397, 410, 288, 233, 258, 183, 325, 387, 182, 307, 357, 314, 322, 92, 304, 323, 210, 93, 464, 94, 255, 271, 15, 219, 259, 3, 294, 359, 96, 209, 336, 265, 319, 360, 272, 277]
# df=pd.read_csv("/home/jixy/naru/datasets/cup.csv")

# df
# for col in df.columns:
#     data=df[col]
#     values=data.value_counts(dropna=False).index.values
#     values=values.tolist()
#     if ('ADATE_2' in values ):
#         break

# cols_old=df.columns

# cols_new = [i for i in range(df.shape[1])]
# assert len(cols_old)==len(cols_new)
# df.columns=cols_new
# df.to_csv("/home/jixy/naru/datasets/cup.csv",index=False)



# import numpy as np
# from tqdm import tqdm
# cols = [
#     'Record Type', 'Registration Class', 'State', 'County', 'Body Type',
#     'Fuel Type', 'Reg Valid Date', 'Color', 'Scofflaw Indicator',
#     'Suspension Indicator', 'Revocation Indicator'
# ]
# df_dmv=pd.read_csv("/home/jixy/naru/datasets/dmv_ofnan.csv",usecols=cols)

# data_len=2000
# df_insert_dmv=pd.DataFrame()
# cols=df_dmv.columns.tolist()
# for i in tqdm(range(data_len)):
#     dict_data={}
#     for j in range(df_dmv.shape[1]):
#         rand_idx_=np.random.randint(0,df_dmv.shape[1],1)[0]
#         dict_data[cols[j]]=df_dmv.iloc[rand_idx_,j]
#     df_insert_dmv=df_insert_dmv.append(dict_data,ignore_index=True)
# df_insert_dmv.to_csv("/home/jixy/naru/datasets/insert_dmv_2000.csv")

# print(df_dmv['Reg Valid Date'][:5])

# type_casts = {'Reg Valid Date': np.datetime64}
# for col, typ in type_casts.items():
#     if col not in df_dmv:
#         continue
#     if typ != np.datetime64:
#         df_dmv[col] = df_dmv[col].astype(typ, copy=False)
#     else:
#         # Both infer_datetime_format and cache are critical for perf.
#         df_dmv[col] = pd.to_datetime(df_dmv[col],
#                                     infer_datetime_format=True,
#                                     cache=True)

# cols = [
#         'Record Type', 'Registration Class', 'State', 'County', 'Body Type',
#         'Fuel Type', 'Reg Valid Date', 'Color', 'Scofflaw Indicator',
#         'Suspension Indicator', 'Revocation Indicator'
#     ]

# df_dmv = df_dmv[cols]
       
# def judge_data(data):

#     if isinstance(data['Reg Valid Date'],pd.Timestamp):
#         return data['Reg Valid Date']
#     else:
#         return pd.Timestamp('2000-01-01')


# df_dmv['Reg Valid Date']=df_dmv.apply(lambda x: judge_data(x), axis=1)
# # df_dmv=df_dmv[df_dmv['Reg Valid Date']!=False]
# df_dmv.to_csv("/home/jixy/naru/datasets/Vehicle__Snowmobile__and_Boat_Registrations_modified.csv")
