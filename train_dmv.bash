#python train_model.py --dataset=dmv --epochs=20 --warmup=20000 --bs=1024     --blocks=4 --dmodel=64 --dff=256 --heads=4 --use_flash_attn >>"/home/jixy/naru/train_log/transformer_flash.log"

python train_model.py --FLASH --epochs=20  --bs=1024 --blocks=2 --flash_dim 64  --dataset=dmv >>"/home/jixy/naru/train_log/FLASH_blocks2_nowarm_dim64.log"

#python train_model.py --FLASH --epochs=20 --warmup=20000 --bs=1024 --blocks=8 --flash_dim 256  --dataset=dmv >>"/home/jixy/naru/train_log/FLASH_blocks8.log"

#python train_model.py --dataset=dmv --epochs=20 --warmups=8000 --bs=2048     --residual --layers=5 --fc-hiddens=256 --direct-io>>"/home/jixy/naru/train_log/resmade.log"