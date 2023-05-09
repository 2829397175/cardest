# python train_model.py --dataset=dmv --epochs=20 --warmup=20000 --bs=1024   \
#   --blocks=4 --dmodel=256 --dff=512 --heads=32 --use_flash_attn \
#   --pretrain /home/jixy/naru/models/pretrained/DMV/DMV_ori_p100 \
#   >>"/home/jixy/naru/train_log/transformer_flash_large.log"

# python train_model.py --FLASH --epochs=20  --bs=1024 --blocks=2 --flash_dim 128  --dataset=dmv >>"/home/jixy/naru/train_log/FLASH_blocks2_nowarm_dim128.log"

# python train_model.py --FLASH --epochs=20 --warmup=20000 --bs=1024 --blocks=4 --flash_dim 256  --dataset=dmv >>"/home/jixy/naru/train_log/FLASH_blocks4_flashdim256.log"

# python train_model.py --blocks=4 --dmodel=128 --dff=256 --heads=8 --dataset cup >>"/home/jixy/naru/train_log/transformer_cup.log"

# python train_model.py --dataset=dmv-ofnan --epochs=20 --warmups=8000 --bs=2048     --residual --layers=5 --fc-hiddens=256 --direct-io \
# --pretrain /home/jixy/naru/models/pretrained/DMV/DMV_ofnan_p100 >>"/home/jixy/naru/train_log/resmade_ofnan.log"

# python train_model.py --dataset=dmv-ofnan --rate_data 0.6 --FLASH --epochs=20  --bs=1024 --blocks=1 --flash_dim 128 --group_size 128 \
# --pretrain /home/jixy/naru/models/pretrained/DMV/DMV_ofnan_p60 >>"/home/jixy/naru/train_log/gau_ofnan_dmvp60.log" >&1

python train_model.py --dataset=dmv-ofnan --rate_data 0.6 --epochs=20 --warmup=20000 --blocks=4 --dmodel=64 --dff=256 --heads=4 --column-masking \
--pretrain /home/jixy/naru/models/pretrained/DMV/DMV_ofnan_p60 >>"/home/jixy/naru/train_log/transformer_dmvp60.log" >&1

# python train_model.py --pretrain /home/jixy/naru/models/pretrained/DMV/DMV_ori_p100 --dataset dmv --epochs=20 --warmup=20000 --bs=1024 --blocks=4 --dmodel=64 --dff=256 --heads=4 --column-masking