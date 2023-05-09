for insert_data in "insert_dmv_2000.csv" "insert_dmv_20000.csv" "insert_dmv_200000.csv"
do

    # transformer
    backbone=/home/jixy/naru/models/pretrained/DMV/DMV_ofnan_p100/dmv-ofnan-1.6MB-model20.172-data19.343-transformer-blocks4-model64-ff256-heads4-use_flash_attnFalse-posEmb-gelu-colmask-20epochs-seed0.pt

    python online_est.py \
    --backbone $backbone --insert_dataset $insert_data --blocks=4 --dmodel=64 --dff=256 --heads=4 --column-masking --num-queries 200 >>"/home/jixy/naru/distill_log/log_distill.txt"

    python online_est.py \
    --backbone $backbone --insert_dataset $insert_data --blocks=4 --dmodel=64 --dff=256 --heads=4 --finetune --column-masking --num-queries 200 >>"/home/jixy/naru/distill_log/log_finetune.txt"

    python online_est.py \
    --backbone $backbone --insert_dataset $insert_data --blocks=4 --dmodel=64 --dff=256 --heads=4 --retrain --column-masking --num-queries 200 >>"/home/jixy/naru/distill_log/log_finetune.txt"




    # gau
    backbone=/home/jixy/naru/models/pretrained/DMV/DMV_ofnan_p100/dmv-ofnan-2.1MB-model20.151-data19.343-flash-blocks1-embed_dim128-expansion_factor2.0-group_size128-posEmb-20epochs-seed0.pt

    python online_est.py \
    --backbone $backbone --insert_dataset $insert_data --FLASH  --blocks=1 --flash_dim 128 --retrain --num-queries 200 >> "/home/jixy/naru/distill_log/logs/gau/log_distill.txt"

    python online_est.py \
    --backbone $backbone --insert_dataset $insert_data --FLASH  --blocks=1 --flash_dim 128  --num-queries 200 >> "/home/jixy/naru/distill_log/logs/gau/log_distill.txt"

    python online_est.py \
    --backbone $backbone --insert_dataset $insert_data --FLASH  --blocks=1 --flash_dim 128 --finetune --num-queries 200 >> "/home/jixy/naru/distill_log/logs/gau/log_distill.txt"


    # backbone=/home/jixy/naru/models/pretrained/DMV/DMV_ofnan_p100/dmv-ofnan-2.1MB-model20.145-data19.343-flash-blocks1-embed_dim128-expansion_factor2.0-group_size2-posEmb-20epochs-seed0.pt

    # python online_est.py \
    # --backbone $backbone --FLASH  --blocks=1 --flash_dim 128 --group_size 2 --num-queries 20 >> "/home/jixy/naru/distill_log/logs/gau/log_distill.txt"

done