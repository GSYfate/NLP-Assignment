MODEL_NAME=t5-base
OUTPUT_DIR=outputs
python train.py \
--model_name $MODEL_NAME \
--dataset_path ag_news \
--output_dir $OUTPUT_DIR \
--max_length 128 \
--epoch_nums 10 \
--train_batch_size 16 \
--val_batch_size 16 \

