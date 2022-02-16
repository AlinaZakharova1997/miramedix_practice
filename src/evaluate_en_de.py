from transformers import MarianMTModel, MarianTokenizer
import torch
import jsonlines
import json
import comet
import os
from comet import download_model, load_from_checkpoint
import nltk
import numpy
import random
from datasets import load_metric, list_metrics
import yaml

params = yaml.safe_load(open('params.yaml'))['evaluate']


model_name_or_path = params['model_name_or_path'] # previously trained and saved model
train_file = params['train_file']
validation_file = params['validation_file']
output_dir = params['output_dir']
source_lang = params['source_lang']
target_lang = params['target_lang']
truncation = params['truncation']
return_tensors = params['return_tensors']
max_length = params['max_length']
skip_special_tokens = params['skip_special_tokens']


BLEU = 'bleu'
ROUGE = 'rouge'
SACREBLEU = 'sacrebleu'
METEOR = 'meteor'
COMET = 'comet'

COMET_MODEL = "wmt21-cometinho-da"
model_path = download_model(COMET_MODEL)
comet_model = load_from_checkpoint(model_path)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_model(model_name):
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name).to(DEVICE)
    return tokenizer, model


def ref_to_bleu(reference_list):
    beam_ref = []
    for item in reference_list:
        text = item[0]
        tokenized = tokenizer.tokenize(text)
        beam_ref.append([tokenized])
    return beam_ref


def hyp_to_bleu(hypothesis_list):
    beam_hyp = []
    for text in hypothesis_list:
        tokenized = tokenizer.tokenize(text)
        beam_hyp.append(tokenized)
    return beam_hyp


def comet_data_maker(source_list, hypothesis_list, reference_list):
    comet_reference = []
    for d in reference_list:
        comet_reference.append(d[0])
        comet_data = [{"src": src, "mt": mt, "ref": ref} for src, mt, ref in zip(source_list, hypothesis_list, comet_reference)]
    return comet_data


def compute_metric(metric_name, references, predictions):
    return load_metric(metric_name).compute(predictions=predictions,
                                            references=references)


def get_results(filename, source_list, hypothesis_list, reference_list):
    path = output_dir+filename
    comet_data = comet_data_maker(source_list, hypothesis_list, reference_list)
    bleu_references = ref_to_bleu(reference_list)
    bleu_predictions = hyp_to_bleu(hypothesis_list)

    results = {BLEU: compute_metric(BLEU, bleu_references, bleu_predictions),
               ROUGE: compute_metric(ROUGE, reference_list, hypothesis_list),
               SACREBLEU: compute_metric(SACREBLEU, reference_list,
                                         hypothesis_list),
               METEOR: compute_metric(METEOR, reference_list, hypothesis_list),
               COMET: comet_model.predict(comet_data, gpus=0),
               }
    with open(path, 'w', encoding='utf8') as f:
        json.dump(results, f)
    return results


with jsonlines.open(validation_file, 'r') as reader:
    target = []
    src = []
    values = []
    for obj in reader:
        values.append(obj)
    for i in range(100):
        random_pair = random.choice(values)
        target.append(random_pair['translation'][target_lang])
        src.append(random_pair['translation'][source_lang])


# Greedy search

source = []
hypothesis = []
reference = []
compare = {}

for i in range(len(target)):
    target_item = target[i]
    src_item = src[i]
    inputs = tokenizer.encode(src_item, return_tensors, max_length, truncation)

    greedy_outputs = model.generate(inputs.cuda())
    result = tokenizer.decode(greedy_outputs[0], skip_special_tokens)
    compare[target_item] = result

    source.append(src_item)
    hypothesis.append(result)
    reference.append([target_item])

get_results('greedy_search_en_de.json', source, hypothesis, reference)

os.system("bash -c '%s'" % script)

