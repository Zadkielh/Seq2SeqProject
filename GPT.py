from transformers import (
    GPT2Tokenizer, GPT2LMHeadModel, TrainerCallback, Trainer, TrainingArguments, DataCollatorForLanguageModeling
)
from datasets import load_dataset
from evaluate import load
import numpy as np
from bs4 import BeautifulSoup
import pandas as pd

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
rouge_metric = load('rouge')


class SaveMetricsCallback(TrainerCallback):
    def __init__(self):
        self.metrics = []

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            metrics['epoch'] = state.epoch
            self.metrics.append(metrics)

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

    soup = BeautifulSoup(text, 'html.parser')
    clean_text = soup.get_text(separator=' ', strip=True)

    clean_text = ' '.join(clean_text.split())

    return clean_text if clean_text else None

def preprocess_function(examples):
    inputs = []
    labels = []
    for question_data, document, annotations in zip(
        examples['question'], examples['document'], examples['annotations']
    ):
        question = question_data if isinstance(question_data, str) else question_data.get('text', '')
        if not question:
            continue

        long_answers = annotations.get('long_answer', [])
        if not isinstance(long_answers, list):
            long_answers = [long_answers]

        long_answer_found = False
        for long_answer in long_answers:
            answer_text = get_long_answer_text(document, long_answer)
            if answer_text:
                input_text = "question: " + question + tokenizer.eos_token + answer_text + tokenizer.eos_token
                encoding = tokenizer(
                    input_text,
                    max_length=128,
                    padding='max_length',
                    truncation=True,
                )
                input_ids = encoding['input_ids']
                attention_mask = encoding['attention_mask']

                try:
                    sep_index = input_ids.index(tokenizer.eos_token_id)
                except ValueError:
                    sep_index = len(input_ids) - 1

                labels_ids = [-100] * (sep_index + 1) + input_ids[sep_index + 1:]
                labels_ids += [-100] * (128 - len(labels_ids))

                inputs.append({
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'labels': labels_ids
                })
                long_answer_found = True
                break

        if not long_answer_found:
            continue

    if inputs:
        model_inputs = {
            'input_ids': [x['input_ids'] for x in inputs],
            'attention_mask': [x['attention_mask'] for x in inputs],
            'labels': [x['labels'] for x in inputs],
        }
        return model_inputs
    else:
        return {}
    
def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    rouge_scores = {key: value * 100 for key, value in result.items()}
    prediction_lens = [len(pred.strip().split()) for pred in decoded_preds]
    rouge_scores['gen_len'] = np.mean(prediction_lens)
    return rouge_scores

def is_non_empty(example):
    return example and 'input_ids' in example and len(example['input_ids']) > 0

if __name__ == '__main__':    
    dataset = load_dataset('natural_questions', 'default')

    train_dataset = dataset['train'].select(range(50000))
    validation_dataset = dataset['validation'].select(range(1000))

    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset['train'].column_names,
        num_proc=4
    )

    validation_dataset = validation_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset['validation'].column_names,
        num_proc=4
    )

    train_dataset = train_dataset.filter(is_non_empty)
    validation_dataset = validation_dataset.filter(is_non_empty)

    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    validation_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    metrics_callback = SaveMetricsCallback()

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        learning_rate=3e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        evaluation_strategy='no',
        save_strategy='epoch',
        save_total_limit=2,
        logging_dir='./logs',
        logging_steps=100,
        fp16=False,
        gradient_accumulation_steps=4,
        eval_accumulation_steps=2
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[metrics_callback],
    )

    trainer.train()

    metrics_df = pd.DataFrame(metrics_callback.metrics)
    metrics_df.to_csv('gpt_evaluation_metrics.csv', index=False)

    results = trainer.evaluate()
    print("Evaluation results:")
    print(results)

    model.save_pretrained('gpt_saved_model')
    tokenizer.save_pretrained('gpt_saved_tokenizer')

    def generate_answer(question):
        input_text = "question: " + question
        input_ids = tokenizer.encode(input_text, return_tensors='pt')
        input_ids = input_ids.to(model.device)
        outputs = model.generate(input_ids, max_new_tokens=50, num_beams=5, early_stopping=True, no_repeat_ngram_size=3, length_penalty=0.8)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer

    sample_question = "What is the tallest mountain in the world?"
    print(f"Question: {sample_question}")
    print(f"Answer: {generate_answer(sample_question)}")

    sample_question = "What is the best topping on pizza?"
    print(f"Question: {sample_question}")
    print(f"Answer: {generate_answer(sample_question)}")

