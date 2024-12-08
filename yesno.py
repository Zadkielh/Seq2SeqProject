import torch
from transformers import (
    BertTokenizer, BertConfig, BertForSequenceClassification, Trainer, TrainingArguments
)
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

dataset = load_dataset('natural_questions', 'default')

def filter_yes_no(example):
    annotations = example['annotations']
    yes_no_answers = annotations.get('yes_no_answer', [])
    return any(answer in [0, 1] for answer in yes_no_answers)

yes_no_train = dataset['train'].filter(filter_yes_no)
yes_no_validation = dataset['validation'].filter(filter_yes_no)

print(f"Number of Yes/No questions in training set: {len(yes_no_train)}")
print(f"Number of Yes/No questions in validation set: {len(yes_no_validation)}")

def extract_labels(dataset):
    labels = []
    for example in dataset:
        annotations = example['annotations']
        yes_no_answers = annotations.get('yes_no_answer', [])
        for answer in yes_no_answers:
            if answer in [0, 1]:
                labels.append(answer)
                break
    return labels

train_labels = extract_labels(yes_no_train)
label_counts = np.bincount(train_labels)
print(f"Label counts: {label_counts}")

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.array([0, 1]),
    y=train_labels
)
class_weights = torch.tensor(class_weights, dtype=torch.float)
print(f"Class Weights: {class_weights}")

class BertForSequenceClassificationWithWeights(BertForSequenceClassification):
    def __init__(self, config, class_weights):
        super().__init__(config)
        self.class_weights = class_weights

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,
            **kwargs
        )
        logits = outputs.logits
        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return {'loss': loss, 'logits': logits}

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def get_long_answer_text(document, long_answer):
    tokens = document.get('tokens')
    if not tokens or 'token' not in tokens:
        return None

    token_list = tokens['token']
    start_token = long_answer['start_token']
    end_token = long_answer['end_token']

    if start_token == -1 or end_token == -1 or start_token >= end_token:
        return None

    end_token = min(end_token, len(token_list))
    selected_tokens = token_list[start_token:end_token]
    text = ' '.join(selected_tokens)
    return text

def preprocess_function(examples):
    inputs = []
    labels = []
    for question_data, document, annotations in zip(
        examples['question'], examples['document'], examples['annotations']
    ):
        question = question_data if isinstance(question_data, str) else question_data.get('text', '')
        if not question:
            continue

        yes_no_answers = annotations.get('yes_no_answer', [])
        long_answers = annotations.get('long_answer', [])
        if not yes_no_answers or not long_answers:
            continue

        for idx, answer in enumerate(yes_no_answers):
            if answer in [0, 1]:
                label = answer
                long_answer = long_answers[idx]
                break
        else:
            continue

        context = get_long_answer_text(document, long_answer)
        if not context:
            continue

        input_text = f"{question} [SEP] {context}"
        inputs.append(input_text)
        labels.append(label)

    if inputs:
        tokenized_inputs = tokenizer(
            inputs, padding='max_length', truncation=True, max_length=512
        )
        tokenized_inputs['labels'] = labels
        return tokenized_inputs
    else:
        return {}

train_dataset = yes_no_train.map(
    preprocess_function,
    batched=True,
    remove_columns=yes_no_train.column_names,
)

validation_dataset = yes_no_validation.map(
    preprocess_function,
    batched=True,
    remove_columns=yes_no_validation.column_names,
)

def filter_empty(example):
    return bool(example.get('input_ids')) and example.get('labels') is not None

train_dataset = train_dataset.filter(filter_empty)
validation_dataset = validation_dataset.filter(filter_empty)

print(f"Number of processed training examples: {len(train_dataset)}")
print(f"Number of processed validation examples: {len(validation_dataset)}")

train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
validation_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

config = BertConfig.from_pretrained('bert-base-uncased', num_labels=2)
model = BertForSequenceClassificationWithWeights.from_pretrained(
    'bert-base-uncased',
    config=config,
    class_weights=class_weights
)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary', zero_division=0
    )
    acc = accuracy_score(labels, preds)
    cm = confusion_matrix(labels, preds)
    print("Confusion Matrix:")
    print(cm)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=5,
    learning_rate=1e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    save_total_limit=2,
    load_best_model_at_end=True,
    logging_dir='./logs',
    logging_steps=100,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

model.save_pretrained('saved_model')
tokenizer.save_pretrained('saved_tokenizer')

from collections import Counter

train_labels = train_dataset['labels'].tolist()
label_counts = Counter(train_labels)
print(f"Label distribution in training set: {label_counts}")