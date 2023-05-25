# activate　naru
script=/home/jixy/naru/eval_model.py

#dmv
# model_path=dmv-6.2MB-model20.311-data19.381-made-resmade-hidden256_256_256_256_256-emb32-directIo-binaryInone_hotOut-inputNoEmbIfLeq-20epochs-seed0.pt
# python $script --model_path "$model_path"  --residual --layers=5 --fc-hiddens=256 --direct-io --dataset=dmv --num-queries 2000

# time-consuming
# model_path=dmv-10.4MB-model20.015-data19.381-flash-blocks4-embed_dim256-expansion_factor2.0-group_size8-posEmb-20epochs-seed0.pt
# python $script --model_path "$model_path"  --FLASH --flash_dim 256  --blocks=4 --group_size 8 --dataset=dmv --run-sampling --run-maxdiff --num-queries 200

# model_path=dmv-2.1MB-model22.003-data19.381-flash-blocks1-embed_dim128-expansion_factor2.0-group_size128-posEmb-20epochs-seed0.pt
# python $script --model_path "$model_path"  --FLASH --blocks=1 --flash_dim 128 --dataset=dmv --num-queries 200 --pretrain /home/jixy/naru/models/pretrained/DMV/DMV_ori_p100

# model_path=dmv-1.6MB-model20.259-data19.381-transformer-blocks4-model64-ff256-heads4-use_flash_attnTrue-posEmb-gelu-colmask-20epochs-seed0.pt
# python $script --model_path "$model_path"  --blocks=4 --dmodel=64 --dff=256 --heads=4 --column-masking --use_flash_attn --dataset=dmv --pretrain /home/jixy/naru/models/pretrained/DMV/DMV_ori_p100 --num-queries 200

# model_path=dmv-11.4MB-model20.123-data19.381-transformer-blocks4-model256-ff512-heads32-use_flash_attnTrue-posEmb-gelu-20epochs-seed0.pt
# python $script --model_path "$model_path"  --blocks=4 --dmodel=256 --dff=512 --heads=32 --use_flash_attn --dataset=dmv --pretrain /home/jixy/naru/models/pretrained/DMV/DMV_ori_p100 --num-queries 2000

# model_path=dmv-100k-1.2MB-model21.272-data16.132-transformer-blocks4-model64-ff256-heads4-use_flash_attnFalse-posEmb-gelu-colmask-20epochs-seed0.pt
# python $script --model_path "$model_path"  --dataset dmv-100k  --blocks=4 --dmodel=64 --dff=256 --heads=4 --column-masking --pretrain /home/jixy/naru/models/pretrained/DMV/DMV_100k_p100 --num-queries 2000

# model_path=dmv-1.6MB-model20.222-data19.381-transformer-blocks4-model64-ff256-heads4-use_flash_attnFalse-posEmb-gelu-colmask-20epochs-seed0.pt
# python $script --model_path "$model_path"  --blocks=4 --dmodel=64 --dff=256 --heads=4 --column-masking --dataset=dmv --pretrain /home/jixy/naru/models/pretrained/DMV/DMV_ori_p100 --num-queries 200


## dmv-ofnan
# model_path=dmv-ofnan-1.6MB-model20.172-data19.343-transformer-blocks4-model64-ff256-heads4-use_flash_attnFalse-posEmb-gelu-colmask-20epochs-seed0.pt
# python $script --model_path "$model_path"  --dataset dmv-ofnan  --blocks=4 --dmodel=64 --dff=256 --heads=4 --column-masking --pretrain /home/jixy/naru/models/pretrained/DMV/DMV_ofnan_p100 --num-queries 2000

# model_path=dmv-ofnan-1.6MB-model20.263-data19.343-transformer-blocks4-model64-ff256-heads4-use_flash_attnTrue-posEmb-gelu-colmask-20epochs-seed0.pt
# python $script --model_path "$model_path"  --dataset dmv-ofnan  --blocks=4 --dmodel=64 --dff=256 --heads=4 --use_flash_attn --column-masking --pretrain /home/jixy/naru/models/pretrained/DMV/DMV_ofnan_p100 --num-queries 2000

# model_path=dmv-ofnan-2.1MB-model20.151-data19.343-flash-blocks1-embed_dim128-expansion_factor2.0-group_size128-posEmb-20epochs-seed0.pt
# python $script --model_path "$model_path"  --dataset dmv-ofnan --FLASH --blocks=1 --flash_dim 128 --num-queries 2000 --pretrain /home/jixy/naru/models/pretrained/DMV/DMV_ofnan_p100 >>/home/jixy/naru/results/logs/dmv_ofnan_gau.txt

# model_path=dmv-ofnan-2.1MB-model20.145-data19.343-flash-blocks1-embed_dim128-expansion_factor2.0-group_size2-posEmb-20epochs-seed0.pt
# python $script --model_path "$model_path"  --dataset dmv-ofnan --FLASH --blocks=1 --flash_dim 128 --group_size 2 --num-queries 2000 --pretrain /home/jixy/naru/models/pretrained/DMV/DMV_ofnan_p100 >>/home/jixy/naru/results/logs/dmv_ofnan_gau_gs2.txt

# model_path=dmv-ofnan-6.0MB-model20.264-data19.343-made-resmade-hidden256_256_256_256_256-emb32-directIo-binaryInone_hotOut-inputNoEmbIfLeq-20epochs-seed0.pt
# python $script --model_path "$model_path"  --residual --layers=5 --fc-hiddens=256 --direct-io --dataset=dmv-ofnan --run-sampling --run-maxdiff --num-queries 2000 --pretrain /home/jixy/naru/models/pretrained/DMV/DMV_ofnan_p100 


# #dmv-tiny
# model_path=dmv-tiny-0.6MB-model10.856-data6.629-made-resmade-hidden128_128_128_128_128-emb32-nodirectIo-binaryInone_hotOut-inputNoEmbIfLeq-20epochs-seed0.pt
# python $script --model_path "$model_path"   --residual --dataset=dmv-tiny --layers=5 --fc-hiddens=128 --num-queries 200

# model_path=dmv-tiny-0.3MB-model5.743-data6.629-transformer-blocks2-model64-ff128-heads4-posEmb-gelu-20epochs-seed0.pt
# python $script --model_path "$model_path"  --dataset=dmv-tiny --blocks=2 --dmodel=64 --dff=128 --heads=4 --use_flash_attn --run-sampling --run-maxdiff --num-queries 200

# model_path=dmv-tiny-0.3MB-model5.467-data6.629-transformer-blocks2-model64-ff128-heads4-use_flash_attnTrue-posEmb-gelu-20epochs-seed0.pt
# python $script --model_path "$model_path"  --dataset=dmv-tiny --blocks=2 --dmodel=64 --dff=128 --heads=4 --use_flash_attn --run-sampling --run-maxdiff --num-queries 200

# model_path=dmv-tiny-0.5MB-model6.804-data6.629-flash-blocks1-embed_dim128-expansion_factor2.0-group_size2-posEmb-20epochs-seed0.pt
# python $script --model_path "$model_path"  --FLASH --flash_dim 128 --group_size 2 --blocks=1 --dataset=dmv-tiny --run-sampling --run-maxdiff --num-queries 200

# #adult
# model_path=adult-0.8MB-model22.269-data15.349-made-resmade-hidden128_128_128_128_128-emb32-nodirectIo-binaryInone_hotOut-inputNoEmbIfLeq-20epochs-seed0.pt
# python $script --model_path "$model_path"   --residual --dataset=adult --layers=5 --fc-hiddens=128 --num-queries 200

# model_path=adult-1.3MB-model27.096-data15.349-transformer-blocks2-model128-ff256-heads8-use_flash_attnTrue-posEmb-gelu-20epochs-seed0.pt
# python $script --model_path "$model_path"  --dataset=adult --blocks=2 --dmodel=128 --dff=256 --heads=8 --use_flash_attn --num-queries 200

# model_path=adult-0.2MB-model22.439-data15.349-transformer-blocks2-model32-ff128-heads4-use_flash_attnFalse-posEmb-gelu-20epochs-seed0.pt
# python $script --model_path "$model_path"  --dataset=adult --heads 4 --blocks 2 --dmodel 32  --num-queries 200

# model_path=adult-1.2MB-model21.331-data15.349-flash-blocks2-embed_dim128-expansion_factor2.0-group_size2-posEmb-20epochs-seed0.pt
# python $script --model_path "$model_path"  --FLASH --flash_dim 128 --group_size 2 --blocks=2 --dataset=adult --num-queries 200 --run-sampling --run-maxdiff


# cup
# model_path=cup-2.9MB-model70.793-data16.542-transformer-blocks4-model128-ff256-heads8-use_flash_attnTrue-posEmb-gelu-20epochs-seed0.pt
# python $script --model_path "$model_path"  --dataset=cup --blocks=4 --dmodel=128 --dff=256 --heads=8 --use_flash_attn --num-queries 200

# model_path=cup-1.7MB-model100.193-data16.542-transformer-blocks2-model128-ff128-heads8-use_flash_attnFalse-posEmb-gelu-20epochs-seed0.pt
# python $script --model_path "$model_path"  --dataset=cup --heads 8 --blocks 2 --dmodel 128 --num-queries 200

# model_path=cup-1.8MB-model91.219-data16.542-flash-blocks2-embed_dim128-expansion_factor2.0-group_size64-posEmb-50epochs-seed0.pt
# python $script --model_path "$model_path"  --FLASH --flash_dim 128 --group_size 64 --blocks=2 --dataset=cup --num-queries 200 --run-sampling --run-maxdiff

model_path=Cup98-4.3MB-model338.530-data16.542-made-resmade-hidden128_128_128_128_128-emb32-directIo-binaryInone_hotOut-inputNoEmbIfLeq-20epochs-seed0.pt
python $script --model_path "$model_path" --dataset=cup \
--residual --layers=5 --fc-hiddens=128 --direct-io --num-queries 200
