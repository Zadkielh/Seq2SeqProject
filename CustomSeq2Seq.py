import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from evaluate import load
import numpy as np
from bs4 import BeautifulSoup
from tqdm import tqdm

from transformers import (
    BertTokenizer,
    GPT2Tokenizer,
)

encoder_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
decoder_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
decoder_tokenizer.pad_token = decoder_tokenizer.eos_token

PAD_IDX = decoder_tokenizer.pad_token_id

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
        if not isinstance(long_answers, list):
            long_answers = [long_answers]

        long_answer_found = False
        for long_answer in long_answers:
            answer_text = get_long_answer_text(document, long_answer)
            if answer_text:
                inputs.append(question)
                targets.append(answer_text)
                long_answer_found = True
                break
        if not long_answer_found:
            continue

    if inputs:
        model_inputs = encoder_tokenizer(
            inputs, padding='max_length', truncation=True, max_length=512
        )

        with decoder_tokenizer.as_target_tokenizer():
            labels = decoder_tokenizer(
                targets, padding='max_length', truncation=True, max_length=150
            )

        decoder_input_ids = [label[:] for label in labels["input_ids"]]

        labels["input_ids"] = [
            [(l if l != decoder_tokenizer.pad_token_id else -100) for l in label]
            for label in labels["input_ids"]
        ]

        model_inputs['labels'] = labels['input_ids']
        model_inputs['decoder_input_ids'] = decoder_input_ids
        return model_inputs
    else:
        return {}

def is_non_empty(example):
    return (
        example and
        'input_ids' in example and len(example['input_ids']) > 0 and
        'labels' in example and len(example['labels']) > 0 and
        'decoder_input_ids' in example and len(example['decoder_input_ids']) > 0
    )

rouge_metric = load('rouge')

def compute_metrics(eval_preds):
    predictions, labels = eval_preds
    preds = np.argmax(predictions, axis=-1)
    labels = np.where(labels != -100, labels, decoder_tokenizer.pad_token_id)

    decoded_preds = decoder_tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = decoder_tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    result = {key: value * 100 for key, value in result.items()}

    prediction_lens = [np.count_nonzero(pred != decoder_tokenizer.pad_token_id) for pred in preds]
    result['gen_len'] = np.mean(prediction_lens)
    return result

class LSTMEncoder(nn.Module):
    def __init__(self, vocab_size, emb_size, hid_size, pad_idx=PAD_IDX):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=pad_idx)
        self.lstm = nn.LSTM(emb_size, hid_size, batch_first=True)
    def forward(self, src):
        embedded = self.embedding(src)
        outputs, (hidden, cell) = self.lstm(embedded)
        return hidden, cell

class LSTMDecoder(nn.Module):
    def __init__(self, vocab_size, emb_size, hid_size, pad_idx=PAD_IDX):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=pad_idx)
        self.lstm = nn.LSTM(emb_size, hid_size, batch_first=True)
        self.fc_out = nn.Linear(hid_size, vocab_size)
    def forward(self, tgt_token, hidden, cell):
        embedded = self.embedding(tgt_token.unsqueeze(1))
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        logits = self.fc_out(output.squeeze(1))
        return logits, hidden, cell

class LSTMSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder, pad_idx=PAD_IDX):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pad_idx = pad_idx

    def forward(self, src, decoder_input_ids, teacher_forcing_ratio=0.5):
        batch_size, tgt_len = decoder_input_ids.size()
        vocab_size = self.decoder.fc_out.out_features
        outputs = torch.zeros(batch_size, tgt_len, vocab_size, device=src.device)
        hidden, cell = self.encoder(src)

        input_token = decoder_input_ids[:, 0]
        for t in range(1, tgt_len):
            logits, hidden, cell = self.decoder(input_token, hidden, cell)
            outputs[:, t, :] = logits
            teacher_force = (torch.rand(1).item() < teacher_forcing_ratio)
            top1 = logits.argmax(1)
            input_token = decoder_input_ids[:, t] if teacher_force else top1
        return outputs

def get_device(model):
    return next(model.parameters()).device

def train_one_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    device = get_device(model)
    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        decoder_input_ids = batch['decoder_input_ids'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        output = model(input_ids, decoder_input_ids, teacher_forcing_ratio=0.5)
        output_dim = output.size(-1)

        out = output[:, 1:].contiguous().view(-1, output_dim)
        lab = labels[:, 1:].contiguous().view(-1)

        loss = criterion(out, lab)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate_model(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    preds_list = []
    labels_list = []
    device = get_device(model)
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            input_ids = batch['input_ids'].to(device)
            decoder_input_ids = batch['decoder_input_ids'].to(device)
            labels = batch['labels'].to(device)

            # No teacher forcing at eval
            output = model(input_ids, decoder_input_ids, teacher_forcing_ratio=0.0)
            output_dim = output.size(-1)
            out = output[:, 1:].contiguous().view(-1, output_dim)
            lab = labels[:, 1:].contiguous().view(-1)
            loss = criterion(out, lab)
            total_loss += loss.item()

            pred_tokens = torch.argmax(output, dim=-1)
            preds_list.append(pred_tokens.cpu())
            labels_list.append(labels.cpu())

    preds = torch.cat(preds_list, dim=0)
    labs = torch.cat(labels_list, dim=0)
    labs_np = labs.numpy()
    labs_np = np.where(labs_np != -100, labs_np, decoder_tokenizer.pad_token_id)
    preds_np = preds.numpy()

    decoded_preds = decoder_tokenizer.batch_decode(preds_np, skip_special_tokens=True)
    decoded_labels = decoder_tokenizer.batch_decode(labs_np, skip_special_tokens=True)
    result = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    result = {key: value * 100 for key, value in result.items()}

    pred_lens = [np.count_nonzero(p != decoder_tokenizer.pad_token_id) for p in preds_np]
    result['gen_len'] = np.mean(pred_lens)

    avg_loss = total_loss / len(dataloader)
    result['eval_loss'] = avg_loss
    return result

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

    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels', 'decoder_input_ids'])
    validation_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels', 'decoder_input_ids'])

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(validation_dataset, batch_size=2)

    EMB_SIZE = 128
    HID_SIZE = 256
    vocab_size_encoder = len(encoder_tokenizer)
    vocab_size_decoder = len(decoder_tokenizer)

    PAD_IDX_ENCODER = encoder_tokenizer.pad_token_id
    PAD_IDX_DECODER = decoder_tokenizer.pad_token_id
    class LSTMEncoder(nn.Module):
        def __init__(self, vocab_size, emb_size, hid_size, pad_idx=PAD_IDX_ENCODER):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=pad_idx)
            self.lstm = nn.LSTM(emb_size, hid_size, batch_first=True)
        def forward(self, src):
            embedded = self.embedding(src)
            outputs, (hidden, cell) = self.lstm(embedded)
            return hidden, cell

    class LSTMDecoder(nn.Module):
        def __init__(self, vocab_size, emb_size, hid_size, pad_idx=PAD_IDX_DECODER):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=pad_idx)
            self.lstm = nn.LSTM(emb_size, hid_size, batch_first=True)
            self.fc_out = nn.Linear(hid_size, vocab_size)
        def forward(self, tgt_token, hidden, cell):
            embedded = self.embedding(tgt_token.unsqueeze(1))
            output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
            logits = self.fc_out(output.squeeze(1))
            return logits, hidden, cell

    encoder = LSTMEncoder(vocab_size_encoder, EMB_SIZE, HID_SIZE, pad_idx=PAD_IDX_ENCODER)
    decoder = LSTMDecoder(vocab_size_decoder, EMB_SIZE, HID_SIZE, pad_idx=PAD_IDX_DECODER)

    class LSTMSeq2Seq(nn.Module):
        def __init__(self, encoder, decoder):
            super().__init__()
            self.encoder = encoder
            self.decoder = decoder
        def forward(self, src, decoder_input_ids, teacher_forcing_ratio=0.5):
            batch_size, tgt_len = decoder_input_ids.size()
            vocab_size = self.decoder.fc_out.out_features
            outputs = torch.zeros(batch_size, tgt_len, vocab_size, device=src.device)
            hidden, cell = self.encoder(src)

            input_token = decoder_input_ids[:, 0]
            for t in range(1, tgt_len):
                logits, hidden, cell = self.decoder(input_token, hidden, cell)
                outputs[:, t, :] = logits
                teacher_force = (torch.rand(1).item() < teacher_forcing_ratio)
                top1 = logits.argmax(1)
                input_token = decoder_input_ids[:, t] if teacher_force else top1
            return outputs

    model = LSTMSeq2Seq(encoder, decoder).to('cuda')

    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    for epoch in range(1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}")

    def generate_answer(model, src_seq, max_len=50):
        model.eval()
        device = get_device(model)
        with torch.no_grad():
            src_seq = src_seq.to(device)
            hidden, cell = model.encoder(src_seq)
            input_token = torch.tensor([decoder_tokenizer.eos_token_id]*src_seq.size(0), device=device)
            outputs = []
            for _ in range(max_len):
                logits, hidden, cell = model.decoder(input_token, hidden, cell)
                top1 = logits.argmax(1)
                outputs.append(top1.item())
                if top1.item() == decoder_tokenizer.eos_token_id:
                    break
                input_token = top1
        return decoder_tokenizer.decode(outputs, skip_special_tokens=True)

    val_results = evaluate_model(model, val_loader, criterion)
    print("Evaluation results:")
    print(val_results)

    sample_question = "What is the tallest mountain in the world?"
    input_ids = encoder_tokenizer.encode(sample_question, return_tensors='pt')
    print(f"Question: {sample_question}")
    print(f"Answer: {generate_answer(model, input_ids)}")

    sample_question = "What is the best topping on pizza?"
    input_ids = encoder_tokenizer.encode(sample_question, return_tensors='pt')
    print(f"Question: {sample_question}")
    print(f"Answer: {generate_answer(model, input_ids)}")