import pandas as pd
import numpy as np
import re



df_infos=[
    {
        "df_name":
        "/home/jixy/naru/distill_results/a0.2_b0.2_results_DMV_ofnan_2000dmv-transformer-blocks4-model64-ff256-heads4-use_flash_attnFalse-posEmb-gelu-colmask.csv",
        "alpha":0.5,
        "beta":0.2,
        "train_epoch":2,
        "type":"distill"
    },
    {
        "df_name":
        "/home/jixy/naru/distill_results/a1.0_b0.2_results_DMV_ofnan_2000dmv-transformer-blocks4-model64-ff256-heads4-use_flash_attnFalse-posEmb-gelu-colmask.csv",
        "alpha":1,
        "beta":0.2,
        "train_epoch":2,
        "type":"distill"
    },
        {
        "df_name":
        "/home/jixy/naru/distill_results/a0.85_b0.2_results_DMV_ofnan_2000dmv-transformer-blocks4-model64-ff256-heads4-use_flash_attnFalse-posEmb-gelu-colmask.csv",
        "alpha":0.85,
        "beta":0.2,
        "train_epoch":2,
        "type":"distill"
    },
                {
        "df_name":
        "/home/jixy/naru/distill_results/a0.8_b0.2_results_DMV_ofnan_2000dmv-transformer-blocks4-model64-ff256-heads4-use_flash_attnFalse-posEmb-gelu-colmask.csv",
        "alpha":0.8,
        "beta":0.2,
        "train_epoch":2,
        "type":"distill"
    },
    {
        "df_name":
        "/home/jixy/naru/distill_results/a0.8_b0.5_results_DMV_ofnan_2000dmv-transformer-blocks4-model64-ff256-heads4-use_flash_attnFalse-posEmb-gelu-colmask.csv",
        "alpha":0.8,
        "beta":0.5,
        "train_epoch":2,
        "type":"distill"
    },
        {
        "df_name":
        "/home/jixy/naru/distill_results/retrain/a1.0_b0.2_results_DMV_ofnan_2000dmv-transformer-blocks4-model64-ff256-heads4-use_flash_attnFalse-posEmb-gelu-colmask_num2000.csv",
        "alpha":1.0,
        "beta":0.2,
        "train_epoch":2,
        "type":"retrain"
    },
    {
        "df_name":
        "/home/jixy/naru/distill_results/a1.0_b.2_opt2_results_DMV_ofnan_2000dmv-transformer-blocks4-model64-ff256-heads4-use_flash_attnFalse-posEmb-gelu-colmask.csv",
        "alpha":1.0,
        "beta":0.2,
        "train_epoch":2,
        "type":"distill"
    },
        {
        "df_name":
        "/home/jixy/naru/distill_results/finetune/finetune_results_DMV_ofnan_2000dmv-transformer-blocks4-model64-ff256-heads4-use_flash_attnFalse-posEmb-gelu-colmask_num20.csv",
        "alpha":None,
        "beta":None,
        "train_epoch":2,
        "type":"finetune"
    },
    {
        "df_name":
        "/home/jixy/naru/distill_results/a0.8_b0.2_results_DMV_ofnan_2000dmv-flash-blocks1-embed_dim128-expansion_factor2.0-group_size128-posEmb.csv",
        "alpha":0.8,
        "beta":0.2,
        "train_epoch":2,
        "type":"distill",
        "group_size":128
    },
    {
        "df_name":
        "/home/jixy/naru/distill_results/a0.8_b0.2_results_DMV_ofnan_2000dmv-flash-blocks1-embed_dim128-expansion_factor2.0-group_size2-posEmb.csv",
        "alpha":0.8,
        "beta":0.2,
        "train_epoch":2,
        "type":"distill",
        "group_size":2
    },
    {
        "df_name":
        "/home/jixy/naru/distill_results/finetune/finetune_results_DMV_ofnan_2000dmv-flash-blocks1-embed_dim128-expansion_factor2.0-group_size128-posEmb_num2000.csv",
        "train_epoch":2,
        "type":"finetune",
        "group_size":128
    },
    {
        "df_name":
        "/home/jixy/naru/distill_results/finetune/finetune_results_DMV_ofnan_2000dmv-transformer-blocks4-model64-ff256-heads4-use_flash_attnFalse-posEmb-gelu-colmask_num2000.csv",
        "train_epoch":2,
        "type":"finetune",
    },
    {
        "df_name":
        "/home/jixy/naru/distill_results/retrain/a1.0_b0.2_results_DMV_ofnan_2000dmv-transformer-blocks4-model64-ff256-heads4-use_flash_attnFalse-posEmb-gelu-colmask_num2000.csv",
        "alpha":1.0,
        "beta":0.2,
        "train_epoch":2,
        "type":"retrain"
    },
        {
        "df_name":
        "/home/jixy/naru/distill_results/retrain/retrain_results_DMV_ofnan_2000dmv-flash-blocks1-embed_dim128-expansion_factor2.0-group_size128-posEmb_num2000.csv",
        "alpha":1.0,
        "beta":0.2,
        "train_epoch":2,
        "type":"retrain"
    },
    {
        "df_name":
        "/home/jixy/naru/distill_results/distill/a0.8_b0.2_1.6MB_results_DMV_ofnan_2000dmv-transformer-blocks4-model64-ff256-heads4-use_flash_attnFalse-posEmb-gelu-colmask_num2000.csv",
        "alpha":0.8,
        "beta":0.2,
        "train_epoch":2,
        "type":"distill"
    }
    ,{
        "df_name":
        "/home/jixy/naru/distill_results/distill/results_DMV_ofnan_2000dmv-flash-blocks1-embed_dim128-expansion_factor2.0-group_size128-posEmb_num2000.csv",
         "alpha":0.8,
        "beta":0.2,
        "train_epoch":2,
        "type":"distill"
    },
    {
        "df_name":
        "/home/jixy/naru/distill_results/distill/results_DMV_ofnan-flash-blocks1-embed_dim128-expansion_factor2.0-group_size128-posEmb_num2000.csv",
         "alpha":0.8,
        "beta":0.2,
        "train_epoch":2,
        "type":"distill"
    }
    
]

new_dataset="distill_result"




def judge_type(name,value):
    if ("maxdiff" in value):
        return "Histogram",None
    elif ("CardEst" in value) or ("psample" in value):
        if ("transformer" in name):
            match = re.search( r'results_(.*?)-transformer', name, re.M|re.I)
            dataset=match.group(1)
            if ("use_flash_attnTrue" in name):
                return "DARE(flash)",dataset
            else:
                return "DARE(transformer)" ,dataset

        elif("flash" in name):
            match = re.search( r'results_(.*?)-flash', name, re.M|re.I)
            dataset=match.group(1)
            return "DARE(GAU)" ,dataset
        elif ("made" in name):
            return "naru(Resmade)" ,None
    elif("sample" in value):
        return "Sampling",None
    else:
        return "error",None
    
    

import os

if __name__=="__main__":        
    df_res=[]
    df_roots=["distill",
              "finetune",
              "retrain"]
    
    
    for type in df_roots:
        df_root="/home/jixy/naru/distill_results/{}".format(type)
        df_names=os.listdir(df_root)
        for df_name in df_names:
            df=pd.read_csv(os.path.join(df_root,df_name))
        
            groups = df.groupby("est")

            for value, group in groups:
                res_d={}
                err=group['err']
                err=np.sort(err)
                len_col=err.shape[0]
                indexs=[int(len_col*0.95),int(len_col*0.5)]
                model_type,dataset=judge_type(df_name,value)
                assert model_type!="error",print(df_name,value)
                res_d['Dataset']=dataset   
                res_d['Model']=model_type    
                res_d['queries']=group.shape[0]
                res_d['Mean']="{:.3f}".format(np.mean(err))
                res_d['Median']="{:.3f}".format(err[indexs[1]])
                res_d['95th']="{:.3f}".format(err[indexs[0]])
                res_d['MAX']="{:.3f}".format(np.max(err))
                res_d['query_ms']="{:.3f}".format(np.mean(group['query_dur_ms']))
                res_d['type']=type
                
                df_res.append(res_d)
        
    # for df_info in df_infos:
    #     df_name=df_info['df_name']
    #     df=pd.read_csv(df_info['df_name'])
        
    #     groups = df.groupby("est")

    #     for value, group in groups:
    #         res_d=df_info
    #         del res_d['df_name']
    #         err=group['err']
    #         err=np.sort(err)
    #         len_col=err.shape[0]
    #         indexs=[int(len_col*0.95),int(len_col*0.5)]
    #         model_type,dataset=judge_type(df_name,value)
    #         assert model_type!="error",print(df_name,value)
    #         res_d['Dataset']=dataset   
    #         res_d['Model']=model_type    
    #         res_d['queries']=group.shape[0]
    #         res_d['Mean']="{:.3f}".format(np.mean(err))
    #         res_d['Median']="{:.3f}".format(err[indexs[1]])
    #         res_d['95th']="{:.3f}".format(err[indexs[0]])
    #         res_d['MAX']="{:.3f}".format(np.max(err))
    #         res_d['query_ms']="{:.3f}".format(np.mean(group['query_dur_ms']))

    #         if ('alpha' in res_d.keys()):
    #             res_d['type']='retrain' if res_d['alpha']==1.0 else res_d['type']
    #         else:
    #             res_d['type']=res_d['type']
    #         df_res.append(res_d)
        
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
    df_res=pd.DataFrame(df_res)
    df_res.to_csv('./distill/tables/{}_card.csv'.format(new_dataset))
    