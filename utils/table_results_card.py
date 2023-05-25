import pandas as pd
import numpy as np
import re

# params
# df_names=[
#     "/home/jixy/naru/results/results_dmv-tiny-0.3MB-model5.743-data6.629-transformer-blocks2-model64-ff128-heads4-posEmb-use_flash_attnTrue-gelu-20epochs-seed0_num200.csv",
#     "/home/jixy/naru/results/results_dmv-tiny-0.3MB-model5.467-data6.629-transformer-blocks2-model64-ff128-heads4-use_flash_attnTrue-posEmb-gelu-20epochs-seed0.csv",
#     "/home/jixy/naru/results/results_dmv-tiny-0.5MB-model6.804-data6.629-flash-blocks1-embed_dim128-expansion_factor2.0-group_size2-posEmb-20epochs-seed0_num200.csv",
#     "/home/jixy/naru/results/results_dmv-tiny-0.6MB-model10.856-data6.629-made-resmade-hidden128_128_128_128_128-emb32-nodirectIo-binaryInone_hotOut-inputNoEmbIfLeq-20epochs-seed0_num200.csv",
    
# ]
# dataset='dmv-tiny'

df_names=[

    "/home/jixy/naru/results/results_dmv-6.2MB-model20.311-data19.381-made-resmade-hidden256_256_256_256_256-emb32-directIo-binaryInone_hotOut-inputNoEmbIfLeq-20epochs-seed0_num2000.csv",
    # "/home/jixy/naru/results/results_dmv-1.6MB-model20.222-data19.381-transformer-blocks4-model64-ff256-heads4-use_flash_attnFalse-posEmb-gelu-colmask-20epochs-seed0.csv",
    # "/home/jixy/naru/results/results_dmv-2.1MB-model22.003-data19.381-flash-blocks1-embed_dim128-expansion_factor2.0-group_size128-posEmb-20epochs-seed0.csv",
    # "/home/jixy/naru/results/results_dmv-10.4MB-model20.015-data19.381-flash-blocks4-embed_dim256-expansion_factor2.0-group_size8-posEmb-20epochs-seed0.csv",
    # "/home/jixy/naru/results/results_dmv-2.6MB-model20.251-data19.381-flash-evenpsampling.csv",
    # "/home/jixy/naru/results/results_dmv-10.4MB-model20.015-data19.381-flash-blocks4-embed_dim256-expansion_factor2.0-group_size8-posEmb-20epochs-seed0_num200.csv",
    # "/home/jixy/naru/results/results_dmv-1.6MB-model20.259-data19.381-transformer-blocks4-model64-ff256-heads4-use_flash_attnTrue-posEmb-gelu-colmask-20epochs-seed0.csv",
    # "/home/jixy/naru/results/results_dmv-11.4MB-model20.123-data19.381-transformer-blocks4-model256-ff512-heads32-use_flash_attnTrue-posEmb-gelu-20epochs-seed0.csv"
    "/home/jixy/naru/results/results_dmv-ofnan-1.6MB-model20.172-data19.343-transformer-blocks4-model64-ff256-heads4-use_flash_attnFalse-posEmb-gelu-colmask-20epochs-seed0.csv",
    "/home/jixy/naru/results/results_dmv-ofnan-1.6MB-model20.263-data19.343-transformer-blocks4-model64-ff256-heads4-use_flash_attnTrue-posEmb-gelu-colmask-20epochs-seed0.csv",
    "/home/jixy/naru/results/results_dmv-ofnan-2.1MB-model20.151-data19.343-flash-blocks1-embed_dim128-expansion_factor2.0-group_size128-posEmb-20epochs-seed0.csv",
    "/home/jixy/naru/results/results_dmv-ofnan-6.0MB-model20.264-data19.343-made-resmade-hidden256_256_256_256_256-emb32-directIo-binaryInone_hotOut-inputNoEmbIfLeq-20epochs-seed0.csv",
    "/home/jixy/naru/results/results_dmv-ofnan-6.0MB-model20.264-data19.343-made-resmade-hidden256_256_256_256_256-emb32-directIo-binaryInone_hotOut-inputNoEmbIfLeq-20epochs-seed0_num2000.csv",
    "/home/jixy/naru/results/results_dmv-ofnan-2.1MB-model20.145-data19.343-flash-blocks1-embed_dim128-expansion_factor2.0-group_size2-posEmb-20epochs-seed0.csv"
]
dataset='dmv'

# df_names=[
#     "/home/jixy/naru/results/results_adult-0.8MB-model22.269-data15.349-made-resmade-hidden128_128_128_128_128-emb32-nodirectIo-binaryInone_hotOut-inputNoEmbIfLeq-20epochs-seed0_num200.csv",
#     "/home/jixy/naru/results/results_adult-0.8MB-model22.269-data15.349-made-resmade-hidden128_128_128_128_128-emb32-nodirectIo-binaryInone_hotOut-inputNoEmbIfLeq-20epochs-seed0.csv",
#     "/home/jixy/naru/results/results_adult-1.2MB-model21.331-data15.349-flash-blocks2-embed_dim128-expansion_factor2.0-group_size2-posEmb-20epochs-seed0_num200.csv",
#     "/home/jixy/naru/results/results_adult-1.2MB-model21.331-data15.349-flash-blocks2-embed_dim128-expansion_factor2.0-group_size2-posEmb-20epochs-seed0.csv",
#     "/home/jixy/naru/results/results_adult-0.2MB-model22.439-data15.349-transformer-blocks2-model32-ff128-heads4-use_flash_attnFalse-posEmb-gelu-20epochs-seed0.csv"
# ]
# dataset='adult'

df_names=[
   "/home/jixy/naru/results/results_cup-1.8MB-model91.219-data16.542-flash-blocks2-embed_dim128-expansion_factor2.0-group_size64-posEmb-50epochs-seed0_num200.csv",
   "/home/jixy/naru/results/results_cup-1.8MB-model91.219-data16.542-flash-blocks2-embed_dim128-expansion_factor2.0-group_size64-posEmb-50epochs-seed0.csv",
   "/home/jixy/naru/results/results_cup-1.7MB-model100.193-data16.542-transformer-blocks2-model128-ff128-heads8-use_flash_attnFalse-posEmb-gelu-20epochs-seed0.csv",
   "/home/jixy/naru/results/results_Cup98-4.3MB-model338.530-data16.542-made-resmade-hidden128_128_128_128_128-emb32-directIo-binaryInone_hotOut-inputNoEmbIfLeq-20epochs-seed0.csv"
]
dataset='cup'



def judge_type(name,value):
    if ("maxdiff" in value):
        return "Histogram"
    elif ("psample" in value):
        if ("transformer" in name):
            if ("use_flash_attnTrue" in name):
                return "DARE(flash)" 
            else:
                return "DARE(transformer)" 
        elif("flash" in name):
            return "DARE(GAU)" 
        elif ("made" in name):
            return "naru(Resmade)" 
    elif("sample" in value):
        return "Sampling"
    else:
        return "error"
    

if __name__=="__main__":        
    df_res=pd.DataFrame()
    for df_name in df_names:
    
        df=pd.read_csv(df_name)
        
        groups = df.groupby("est")

        for value, group in groups:
            res_d={}
            err=group['err']
            err=np.sort(err)
            len_col=err.shape[0]
            indexs=[int(len_col*0.95),int(len_col*0.5)]
            model_type=judge_type(df_name,value)
            assert model_type!="error",print(df_name,value)
            
            res_d['Model']=model_type
            if (model_type!="Histogram" and model_type!="Histogram"):
                match3 = re.search( r'(\d+\.?\d+)MB', df_name, re.M|re.I)
                size_f=float(match3.group(1))
                res_d['Size']="{:.3f}".format(size_f)+"MB"
            else:
                res_d['Size']="N/A"
            
            res_d['Mean']="{:.3f}".format(np.mean(err))
            res_d['Median']="{:.3f}".format(err[indexs[1]])
            res_d['95th']="{:.3f}".format(err[indexs[0]])
            res_d['MAX']="{:.3f}".format(np.max(err))
            res_d['query_ms']="{:.3f}".format(np.mean(group['query_dur_ms']))

            df_res=df_res.append(res_d,ignore_index=True)
        
        # groups = df_res.groupby("Model")
        # df_new=[]
        # for value,df_sub in groups:
        #     groups_sub=df_sub.groupby("Size")
        #     if (value in ["Sampling","Histogram"]):
        #         index_max=np.argmax(df_sub['MAX'])
        #         df_new.append(pd.DataFrame([df_sub.iloc[index_max,:]]))
        #         continue
        #     for value_size,df_item in groups_sub:
        #         if ("DARE" in value):
        #             index_min=np.argmin(df_item['MAX'])
        #             df_new.append(pd.DataFrame([df_item.iloc[index_min,:]]))
        #         else:
        #             index_max=np.argmax(df_item['MAX'])
        #             df_new.append(pd.DataFrame([df_item.iloc[index_max,:]]))
        # df_res=pd.concat(df_new)
    df_res.to_csv('./tables/{}_card.csv'.format(dataset))
    