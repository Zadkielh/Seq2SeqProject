from transformers import (
    BartTokenizer, BartForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments, TrainerCallback
)
from datasets import load_dataset
from evaluate import load
import numpy as np
from bs4 import BeautifulSoup
import pandas as pd

tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')

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
    targets = []
    for question_data, document, annotations in zip(
        examples['question'], examples['document'], examples['annotations']
    ):
        question = question_data if isinstance(question_data, str) else question_data.get('text', '')
        if not question:
            continue

        long_answers = annotations.get('long_answer', [])
        long_answer_found = False
        for idx, long_answer in enumerate(long_answers):
            answer_text = get_long_answer_text(document, long_answer)
            if answer_text:
                inputs.append("question: " + question)
                targets.append(answer_text)
                long_answer_found = True
                break
        if not long_answer_found:
            continue

    if inputs:
        model_inputs = tokenizer(
            inputs, padding='max_length', truncation=True, max_length=512
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets, padding='max_length', truncation=True, max_length=150
            )

        model_inputs['labels'] = labels['input_ids']
        return model_inputs
    else:
        return {}
    

def compute_metrics(eval_preds):
    predictions, labels = eval_preds
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    rouge_scores = {key: value * 100 for key, value in result.items()}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    rouge_scores['gen_len'] = np.mean(prediction_lens)
    return rouge_scores

def is_non_empty(example):
    return example and 'input_ids' in example and len(example['input_ids']) > 0

if __name__ == '__main__':
    dataset = load_dataset('natural_questions', 'default')

    train_dataset = dataset['train'].select(range(50000))
    validation_dataset = dataset['validation'].select(range(5000))

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

    training_args = Seq2SeqTrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        learning_rate=3e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        eval_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=2,
        predict_with_generate=True,
        logging_dir='./logs',
        logging_steps=100,
        fp16=False,
        gradient_accumulation_steps=4
    )

    metrics_callback = SaveMetricsCallback()

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[metrics_callback]
    )

    trainer.train()

    metrics_df = pd.DataFrame(metrics_callback.metrics)
    metrics_df.to_csv('bart_evaluation_metrics.csv', index=False)

    results = trainer.evaluate()
    print("Evaluation results:")
    print(results)

    model.save_pretrained('bart_saved_model')
    tokenizer.save_pretrained('bart_saved_tokenizer')

    def generate_answer(question):
        input_text = "question: " + question
        input_ids = tokenizer.encode(input_text, return_tensors='pt')
        input_ids = input_ids.to(model.device)
        outputs = model.generate(input_ids, max_new_tokens=20, num_beams=5, early_stopping=True, no_repeat_ngram_size=3, length_penalty=0.8)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer

    sample_question = "What is the tallest mountain in the world?"
    print(f"Question: {sample_question}")
    print(f"Answer: {generate_answer(sample_question)}")

    sample_question = "What is the best topping on pizza?"
    print(f"Question: {sample_question}")
    print(f"Answer: {generate_answer(sample_question)}")
