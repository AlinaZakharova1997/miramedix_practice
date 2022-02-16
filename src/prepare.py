import os
import jsonlines
import yaml
import random

params = yaml.safe_load(open("params.yaml"))["prepare"]

raw_data_file = ["raw_data_file"]
random.seed(params["seed"])
train_size = params["train_size"]
test_size = params["test_size"]
val_size = params["val_size"]

data = []

with jsonlines.open("data/source.json", 'r') as reader:
    for obj in reader:
        data.append(obj)


def divide(data, train_size, val_size, test_size):
    train = data[:train_size]
    val = data[train_size:train_size+val_size]
    test = data[train_size+val_size:train_size+val_size+test_size]
    return train, val, test


train, val, test = divide(data, train_size, val_size, test_size)


os.mkdir('data/prepared')

with jsonlines.open('data/prepared/test.jsonl', mode='w') as writer:
    writer.write_all(test)
with jsonlines.open('data/prepared/val.jsonl', mode='w') as writer:
    writer.write_all(val)
with jsonlines.open('data/prepared/train.jsonl', mode='w') as writer:
    writer.write_all(train)

