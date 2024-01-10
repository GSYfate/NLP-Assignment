MODEL_NAME=t5-small
OUTPUT_DIR=outputs
python eval.py \
--model_name $MODEL_NAME \
--dataset_path ag_news \
--output_dir $OUTPUT_DIR \
--max_length 128 \
--val_batch_size 32 \
--checkpoint_path outputs/model_checkpoint_epoch_4.pt