TRAIN_FILE=/mnt/data10t/bakuphome20210617/ljh/github/Corpus/mergednews.181920.txt
export CUDA_VISIBLE_DEVICES=5,6,7

nohup python mlm.py  \
    --do_train \
    --train_file $TRAIN_FILE \
    --do_eval \
    --line_by_line \
    --max_seq_length 512 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --num_train_epochs 10\
    --save_strategy epoch \
    --overwrite_output_dir \
    --output_dir=./checkpoints 2>&1 >mlm1022.log &