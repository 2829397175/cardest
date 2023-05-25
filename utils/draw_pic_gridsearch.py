
import matplotlib.pyplot as plt
import pandas as pd


df=pd.read_csv("/home/jixy/naru/params/params_gridsearch_newdata.csv")

df=df[df['start_lossw']!=1]

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

err=df['err']-33.6
plt.scatter(df['start_lossw'],err,color='r',label = "beta == 0.2")           
    
plt.ylabel('entropy gap')
plt.xlabel('alpha')
plt.legend(fontsize=16)        #个性化图例（颜色、形状等）
plt.savefig("./distill/pics/beta0.2.png") #保存图片 路径：/imgPath/