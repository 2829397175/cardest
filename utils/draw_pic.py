
from tkinter import font
import matplotlib.pyplot as plt
import pandas as pd
# params

# dmv-tiny
df_names=[
    "/home/jixy/naru/train_log/dmv-tiny-0.6MB-model10.856-data6.629-made-resmade-hidden128_128_128_128_128-emb32-nodirectIo-binaryInone_hotOut-inputNoEmbIfLeq-20epochs-seed0.csv",
    # "/home/jixy/naru/train_log/dmv-tiny-0.3MB-model5.467-data6.629-transformer-blocks2-model64-ff128-heads4-use_flash_attnTrue-posEmb-gelu-20epochs-seed0.csv",
    "/home/jixy/naru/train_log/dmv-tiny-0.5MB-model6.804-data6.629-flash-blocks1-embed_dim128-expansion_factor2.0-group_size2-posEmb-20epochs-seed0.csv",
    "/home/jixy/naru/train_log/dmv-tiny-0.8MB-model14.547-data6.629-transformer-blocks2-model128-ff128-heads32-use_flash_attnFalse-posEmb-gelu-2epochs-seed0.csv"
]
dataset='dmv-tiny'
dataset_entropy=6.629

# # dmv
# df_names=[
#     "/home/jixy/naru/train_log/dmv-ofnan-1.6MB-model20.172-data19.343-transformer-blocks4-model64-ff256-heads4-use_flash_attnFalse-posEmb-gelu-colmask-20epochs-seed0.csv",
#     "/home/jixy/naru/train_log/dmv-ofnan-2.1MB-model20.151-data19.343-flash-blocks1-embed_dim128-expansion_factor2.0-group_size128-posEmb-20epochs-seed0.csv",
#     "/home/jixy/naru/train_log/dmv-6.2MB-model20.311-data19.381-made-resmade-hidden256_256_256_256_256-emb32-directIo-binaryInone_hotOut-inputNoEmbIfLeq-20epochs-seed0.csv",
# ]
# dataset='dmv'
# dataset_entropy=19.381

# #cup
# df_names=[
# "/home/jixy/naru/train_log/cup-1.8MB-model94.557-data16.542-flash-blocks2-embed_dim128-expansion_factor2.0-group_size64-posEmb-20epochs-seed0.csv",
# # "/home/jixy/naru/train_log/cup-2.9MB-model70.793-data16.542-transformer-blocks4-model128-ff256-heads8-use_flash_attnTrue-posEmb-gelu-20epochs-seed0.csv",
# "/home/jixy/naru/train_log/cup-2.9MB-model100.605-data16.542-transformer-blocks4-model128-ff256-heads8-use_flash_attnFalse-posEmb-gelu-20epochs-seed0.csv",

# ]

# dataset='cup98'
# dataset_entropy=16.542

# # # adult
# df_names=[
# "/home/jixy/naru/train_log/adult-0.8MB-model22.269-data15.349-made-resmade-hidden128_128_128_128_128-emb32-nodirectIo-binaryInone_hotOut-inputNoEmbIfLeq-20epochs-seed0.csv",
# "/home/jixy/naru/train_log/adult-1.2MB-model21.331-data15.349-flash-blocks2-embed_dim128-expansion_factor2.0-group_size2-posEmb-20epochs-seed0.csv",
# # "/home/jixy/naru/train_log/adult-1.3MB-model27.096-data15.349-transformer-blocks2-model128-ff256-heads8-use_flash_attnTrue-posEmb-gelu-20epochs-seed0.csv",
# # "/home/jixy/naru/train_log/adult-1.3MB-model27.096-data15.349-transformer-blocks2-model128-ff256-heads8-use_flash_attnTrue-posEmb-gelu-20epochs-seed0.csv",

# "/home/jixy/naru/train_log/adult-0.2MB-model22.439-data15.349-transformer-blocks2-model32-ff128-heads4-use_flash_attnFalse-posEmb-gelu-20epochs-seed0.csv"
# ]

# dataset='census(adult)'
# dataset_entropy=15.349

def judge_type(name):
    if ("transformer" in name):
        if ("use_flash_attnTrue" in name):
            return "DARE(flash)",'blue'
        else:
            return "DARE(Transformer)",'y'
    elif("flash" in name):
        return "DARE(GAU)",'r'
    elif ("made" in name):
        return "naru(ResMade)",'g'
    



    
plt.figure()                   # 设置图片信息 例如：plt.figure(num = 2,figsize=(640,480))
import numpy as np
for df_name in df_names:
    df=pd.read_csv(df_name,index_col=0)
    epochs=[i+1 for i in range(10)]
    model_type,color=judge_type(df_name)
    df['loss']/=np.log(2)
    data=df['loss']-dataset_entropy
    data=data[:10]
    plt.plot(epochs,data,color=color,label = model_type)           
    

plt.ylabel('entropy gap',fontsize=16)
plt.xlabel('epoch',fontsize=16)
plt.legend(fontsize=16)        #个性化图例（颜色、形状等）
plt.savefig("./pics/{}_train_loss.png".format(dataset)) #保存图片 路径：/imgPath/