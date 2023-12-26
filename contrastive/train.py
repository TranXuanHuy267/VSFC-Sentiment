
import os
import numpy  as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer,BertForSequenceClassification
from roberta import RobertaForSequenceClassification
from datasets import load_dataset, load_metric
from datasets import ClassLabel
from datasets import load_dataset, load_metric
import pandas as pd
import torch
from transformers import BertTokenizer
import numpy as np
from scipy.special import softmax
from transformers import AutoTokenizer
train = pd.read_csv("/home4/thanhpv/ser/ABSA/prepared_train.csv")
valid = pd.read_csv("/home4/thanhpv/ser/ABSA/prepared_valid.csv")
test = pd.read_csv("/home4/thanhpv/ser/ABSA/prepared_test.csv")
num_class = 3
device="cuda"
train_dataset = load_dataset("csv", data_files="train.csv", sep="|",split="train")
test_dataset = load_dataset("csv", data_files="dev.csv",sep="|", split="train")



def speech_file_to_array_fn(batch):
    text= batch["sentence"]
    batch["text"] = text
    batch["labels"]=batch["label"]
    return batch

train_dataset = train_dataset.map(speech_file_to_array_fn, remove_columns=train_dataset.column_names)
test_dataset = test_dataset.map(speech_file_to_array_fn, remove_columns=test_dataset.column_names)
print(train_dataset)
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets_train = train_dataset.map(tokenize_function, batched=True)
tokenized_datasets_test=test_dataset.map(tokenize_function, batched=True)

# small_train_dataset = tokenized_datasets_train.shuffle(seed=42).select(range(1000))
# small_eval_dataset  = tokenized_datasets_test.shuffle(seed=42).select(range(1000))

full_train_dataset  = tokenized_datasets_train
full_eval_dataset   = tokenized_datasets_test

metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

model         = RobertaForSequenceClassification.from_pretrained("vinai/phobert-base", num_labels=num_class)
# training_args = TrainingArguments("test_trainer")
training_args = TrainingArguments("checkpoint_fusion", 
  per_device_train_batch_size=32,
  evaluation_strategy="steps",
  num_train_epochs=30,
  save_steps=200,
  eval_steps=200,
  logging_steps=50,
  load_best_model_at_end=True,
  dataloader_num_workers=6,
  learning_rate=2e-5,
  weight_decay=0.01,
  save_total_limit=3,
  report_to='tensorboard')
trainer       = Trainer(
    model         = model, 
    args          = training_args, 
    train_dataset = full_train_dataset, 
    eval_dataset  = full_eval_dataset,
    compute_metrics = compute_metrics,
)
trainer.train()
trainer.evaluate()
