from this import d
import pandas as pd

df_names=[
    "results_dmv-tiny-0.6MB-model10.856-data6.629-made-resmade-hidden128_128_128_128_128-emb32-nodirectIo-binaryInone_hotOut-inputNoEmbIfLeq-20epochs-seed0.csv",
    "results_adult-1.2MB-model21.331-data15.349-flash-blocks2-embed_dim128-expansion_factor2.0-group_size2-posEmb-20epochs-seed0.csv",
    "results_adult-1.3MB-model27.096-data15.349-transformer-blocks2-model128-ff256-heads8-use_flash_attnTrue-posEmb-gelu-20epochs-seed0.csv",
    "results_dmv-tiny-0.3MB-model5.743-data6.629-transformer-blocks2-model64-ff128-heads4-posEmb-gelu-20epochs-seed0.csv",
    "results_adult-1.3MB-model27.096-data15.349-transformer-blocks2-model128-ff256-heads8-use_flash_attnTrue-posEmb-gelu-20epochs-seed0.csv",
    "results_dmv-tiny-0.3MB-model5.743-data6.629-transformer-blocks2-model64-ff128-heads4-posEmb-gelu-20epochs-seed0.csv",
    "results_adult-1.3MB-model27.096-data15.349-transformer-blocks2-model128-ff256-heads8-use_flash_attnTrue-posEmb-gelu-20epochs-seed0.csv",
    "results_dmv-tiny-0.3MB-model5.743-data6.629-transformer-blocks2-model64-ff128-heads4-posEmb-gelu-20epochs-seed0.csv",
    "results_dmv-tiny-0.5MB-model6.804-data6.629-flash-blocks1-embed_dim128-expansion_factor2.0-group_size2-posEmb-20epochs-seed0.csv",
    "results_adult-1.3MB-model27.096-data15.349-transformer-blocks2-model128-ff256-heads8-use_flash_attnTrue-posEmb-gelu-20epochs-seed0.csv",
    "results_adult-0.8MB-model22.269-data15.349-made-resmade-hidden128_128_128_128_128-emb32-nodirectIo-binaryInone_hotOut-inputNoEmbIfLeq-20epochs-seed0.csv",
    "results_dmv-6.2MB-model20.311-data19.381-made-resmade-hidden256_256_256_256_256-emb32-directIo-binaryInone_hotOut-inputNoEmbIfLeq-20epochs-seed0.csv",
    "results_dmv-10.4MB-model20.015-data19.381-flash-blocks4-embed_dim256-expansion_factor2.0-group_size8-posEmb-20epochs-seed0.csv",
    "results_cup-1.8MB-model91.219-data16.542-flash-blocks2-embed_dim128-expansion_factor2.0-group_size64-posEmb-50epochs-seed0.csv",
    "results_dmv-6.2MB-model20.311-data19.381-made-resmade-hidden256_256_256_256_256-emb32-directIo-binaryInone_hotOut-inputNoEmbIfLeq-20epochs-seed0_num2000.csv",
    "results_dmv-10.4MB-model20.015-data19.381-flash-blocks4-embed_dim256-expansion_factor2.0-group_size8-posEmb-20epochs-seed0_num200.csv",
    "results_cup-2.9MB-model70.793-data16.542-transformer-blocks4-model128-ff256-heads8-use_flash_attnTrue-posEmb-gelu-20epochs-seed0.csv",
    "results_cup-1.8MB-model91.219-data16.542-flash-blocks2-embed_dim128-expansion_factor2.0-group_size64-posEmb-50epochs-seed0_num200.csv",
]

import os
root_dir="/home/jixy/naru/results"
for df_name in df_names:
    df=pd.read_csv(os.path.join(root_dir,df_name))
    groups = df.groupby("est")
    new_dfs=[]
    for value, group in groups:
        if ("psample" in value):
            group['query_dur_ms']/=2
        new_dfs.append(group)
    df=pd.concat(new_dfs)
    df.to_csv(os.path.join(root_dir,df_name))
    
