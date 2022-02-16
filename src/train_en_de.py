import os

import yaml

from transformers.file_utils import hf_bucket_url, cached_path

params = yaml.safe_load(open('params.yaml'))['train']

model_name_or_path = params['model_name_or_path']
do_train = params['do_train']
do_eval = params['do_eval']
save_strategy = params['save_strategy']
source_lang = params['source_lang']
target_lang = params['target_lang']
num_train_epochs = params['num_train_epochs']
max_source_length = params['max_source_length']
max_target_length = params['max_target_length']
val_max_target_length = params['val_max_target_length']
train_file = params['train_file']
output_dir = params['output_dir']
per_device_train_batch_size = params['per_device_train_batch_size']
overwrite_output_dir = params['overwrite_output_dir']
pad_to_max_length = params['pad_to_max_length']
logging_steps = params['logging_steps']
save_steps = params['save_steps']

script = f"""
python3 transformers/examples/pytorch/translation/run_translation.py \
--model_name_or_path {model_name_or_path} \
--do_train {do_train} \
--do_eval {do_eval} \
--save_strategy {save_strategy} \
--source_lang {source_lang} \
--target_lang {target_lang} \
--num_train_epochs {num_train_epochs} \
--max_source_length {max_source_length} \
--max_target_length {max_target_length} \
--train_file {train_file} \
--output_dir {output_dir} \
--per_device_train_batch_size {per_device_train_batch_size} \
--overwrite_output_dir {overwrite_output_dir} \
--pad_to_max_length {pad_to_max_length} \
--logging_steps {logging_steps} \
--save_steps {save_steps} 
"""
os.mkdir('trained_model')
os.system("bash -c '%s'" % script)
