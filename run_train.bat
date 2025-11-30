@echo off
set KMP_DUPLICATE_LIB_OK=TRUE
call conda activate chapter-llama
python train.py data=highlight paths.output_dir=outputs/highlight_test model.config_train.model_name=D:/chapter-llama/Llama-3.2-1B-Instruct model.config_train.num_epochs=1 model.config_train.batch_size_training=1 model.config_train.gradient_accumulation_steps=1 model.config_train.context_length=4096 model.config_train.use_peft=False model.config_train.output_dir=outputs/highlight_test/model model.config_train.enable_fsdp=False > train_log.txt 2>&1
echo Training completed. Check train_log.txt for details.
pause
