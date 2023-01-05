# All code related to T5 finetuning adapted from: https://colab.research.google.com/github/abhimishra91/transformers-tutorials/blob/master/transformers_summarization_wandb.ipynb#scrollTo=932p8NhxeNw4
# Also referred to: https://github.com/priya-dwivedi/Deep-Learning/blob/master/wikihow-fine-tuning-T5/Tune_T5_WikiHow-Github.ipynb
# Finetune T5 or UnifiedQA on TellMeWhy in various settings

import torch
import argparse
import numpy as np
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Model, AdamW
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
import os
import sys
import csv
import logging
import random
import data
import json

np.random.seed(1234)
torch.manual_seed(1234)
random.seed(1234)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, help='Supports train and test mode')
    parser.add_argument('--model-arch', type=str, help='Model architecture to use', default='t5-base')
    parser.add_argument('--input-type', type=str, help='Specify type of input format; normal, needs-effects')
    parser.add_argument('--data-folder', type=str, help='Folder with all the data from the corpus')
    parser.add_argument('--knowledge-folder', type=str, help='Folder with COMET data dictionaries')
    parser.add_argument('--dataset-name', type=str, help='Name of dataset in use')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.00005, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0, help='Weight decay')
    parser.add_argument('--eps', type=float, default=1e-8, help='AdamW epsilon')
    parser.add_argument('--model-dir', type=str, help='Directory to save model information in or load it from (for testing)')
    parser.add_argument('--log-file', type=str, required=True, help='Log filepath to write training information to')
    parser.add_argument('--source-len', type=int, default=150, help='Length of input context')
    parser.add_argument('--answer-len', type=int, default=30, help='Desired length of answer')
    parser.add_argument('--patience', type=int, default=3, help='Early stopping criteria')
    parser.add_argument('--test-output-file', type=str, help='Test output file')
    parser.add_argument('--hidden-test', dest='hidden_test', action='store_true', help='Set true if testing on hidden test set')
    args, _ = parser.parse_known_args()
    return args

def create_model_config(args):

    """
    This function creates model configuration dictionary and save it in the model directory.
    It is only called when model goes into training mode.
    """

    config = {
        "dataset_name": args.dataset_name,
        "model_input_type": args.input_type,
        "save_model_dir": args.model_dir,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "source_length": args.source_len,
        "answer_length": args.answer_len
    }

    json.dump(config, open(os.path.join(args.model_dir, "model_config.json"), 'w+'))

class T5Dataset(Dataset):

    def __init__(self, dataframe, tokenizer, context_len, answer_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.context_len = context_len
        self.answer_len = answer_len
        self.answer = self.data['answer']
        self.context = self.data['model_input']

    def __len__(self):
        return len(self.answer)

    def __getitem__(self, index):
        context = str(self.context[index])
        context = ' '.join(context.split())

        answer = str(self.answer[index])
        answer = ' '.join(answer.split())

        source = self.tokenizer.batch_encode_plus([context], max_length=self.context_len, padding='max_length', truncation=True, return_tensors='pt')
        target = self.tokenizer.batch_encode_plus([answer], max_length=self.answer_len, padding='max_length', truncation=True, return_tensors='pt')

        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        target_mask = target['attention_mask'].squeeze()

        return {
            'source_ids': source_ids.to(dtype=torch.long), 
            'source_mask': source_mask.to(dtype=torch.long), 
            'target_ids': target_ids.to(dtype=torch.long),
            'target_ids_y': target_ids.to(dtype=torch.long)
        }

def train(tokenizer, model, device, loader, optimizer):

    """
    This function runs train step of fine-tuning for T5.
    """

    model.train()
    total_train_loss = []
    for _, data in tqdm(enumerate(loader)):
        y = data['target_ids'].to(device)
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone().detach()
        lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
        lm_labels = lm_labels.to(device)
        ids = data['source_ids'].to(device)
        mask = data['source_mask'].to(device)

        model.zero_grad()
        outputs = model(input_ids = ids, attention_mask = mask, decoder_input_ids=y_ids, labels=lm_labels)
        loss = outputs[0]
        
        loss.backward()
        total_train_loss.append(loss.data.cpu().numpy().tolist())
        optimizer.step()
        torch.cuda.empty_cache()
    train_loss = sum(total_train_loss) / len(total_train_loss)
    return train_loss

def evaluate(tokenizer, model, device, loader):

    """
    This function runs evaluate step of fine-tuning for T5.
    """

    model.eval()
    total_dev_loss = []
    with torch.no_grad():
        for _, data in tqdm(enumerate(loader)):
            y = data['target_ids'].to(device)
            y_ids = y[:, :-1].contiguous()
            lm_labels = y[:, 1:].clone().detach()
            lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
            lm_labels = lm_labels.to(device)
            ids = data['source_ids'].to(device)
            mask = data['source_mask'].to(device)

            outputs = model(input_ids = ids, attention_mask = mask, decoder_input_ids=y_ids, labels=lm_labels)
            loss = outputs[0]
            
            total_dev_loss.append(loss.data.cpu().numpy().tolist())
        dev_loss = sum(total_dev_loss) / len(total_dev_loss)
    return dev_loss

def finetune(train_dataset, val_dataset, model, tokenizer, device, args):

    """
    This function handles all the steps for fine-tuning T5
    """

    training_set = T5Dataset(train_dataset, tokenizer, args.source_len, args.answer_len)
    val_set = T5Dataset(val_dataset, tokenizer, args.source_len, args.answer_len)

    train_params = {
        'batch_size': args.batch_size,
        'shuffle': True,
        'num_workers': 0
        }

    val_params = {
        'batch_size': args.batch_size,
        'shuffle': False,
        'num_workers': 0
        }

    training_loader = DataLoader(training_set, **train_params)
    val_loader = DataLoader(val_set, **val_params)

    logging.info('Data loaders ready')

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.eps)

    best_dev_loss = 100000
    best_bertscore = -9999
    patience = 0

    log_str = f'Number of train batches: {len(training_loader)}; Number of dev batches: {len(val_loader)}'
    logging.info(log_str)

    model.to(device)
    for epoch in range(1, args.epochs+1):
        train_loss = train(tokenizer, model, device, training_loader, optimizer)
        patience += 1
        val_loss = evaluate(tokenizer, model, device, val_loader)
        log_str = f'Epoch: {epoch} | Train loss: {train_loss} | Dev loss: {val_loss}'
        logging.info(log_str)
        if val_loss < best_dev_loss:
            log_str = f'Best model at epoch {epoch}; Saving...'
            logging.info(log_str)
            best_dev_loss = val_loss
            torch.save(model, os.path.join(args.model_dir, 'model.pt'))
            patience = 0
        if patience >= args.patience:
            log_str = f'Stopping criterion (dev loss) hasn\'t improved for {args.patience} epochs; Stopping training...'
            logging.info(log_str)
            logging.info('Done')
            sys.exit()
    logging.info('Done')
    sys.exit()

def finetuned_model_inference(test_dataset, model, tokenizer, device, args):

    """
    This function takes a pytorch T5 model (huggingface model that has been finetuned) and performs inference for a given test set.
    """

    test_set = T5Dataset(test_dataset, tokenizer, args.source_len, args.answer_len)
    test_params = {
        'batch_size': args.batch_size,
        'shuffle': False,
        'num_workers': 0
        }
    test_loader = DataLoader(test_set, **test_params)

    log_str = f'Number of test batches: {len(test_loader)}'
    logging.info(log_str)

    model.to(device)
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for _, data in tqdm(enumerate(test_loader)):
            y = data['target_ids'].to(device)
            ids = data['source_ids'].to(device)
            mask = data['source_mask'].to(device)

            generated_ids = model.generate(
                input_ids = ids,
                attention_mask = mask, 
                max_length=args.answer_len, 
                num_beams=5,
                repetition_penalty=2.5, 
                length_penalty=1.0, 
                early_stopping=True
                )
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in y]

            predictions.extend(preds)
            actuals.extend(target)

    assert len(predictions) == len(actuals)
    assert test_dataset.shape[0] == len(predictions)
    return predictions, actuals

def main(args):
    logging.basicConfig(filename=args.log_file, level=logging.DEBUG, format='%(asctime)s %(message)s')
    logging.info(vars(args))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        x = torch.rand((3,5))
        x = x.to(device)
        torch.backends.cudnn.deterministic = True
        logging.info('Placed nominal tensor onto GPU to reserve it')
    else:
        logging.info('Using cpu; did you set CUDA_VISIBLE_DEVICES in order to use a gpu?')

    if args.dataset_name == 'tellmewhy':
        train_df, dev_df, test_df, hidden_test_df = data.load_tellmewhy(args.data_folder)

    tokenizer = T5Tokenizer.from_pretrained(args.model_arch)

    if args.input_type == 'normal':
        train_df = train_df.apply(data.create_normal_model_input, axis=1)
        dev_df = dev_df.apply(data.create_normal_model_input, axis=1)

        test_df = test_df.apply(data.create_normal_model_input, axis=1)
        hidden_test_df = hidden_test_df.apply(data.create_normal_model_input, axis=1)
        
    elif args.input_type == 'separator-normal':
        train_df = train_df.apply(data.create_separator_model_input, axis=1)
        dev_df = dev_df.apply(data.create_separator_model_input, axis=1)

        test_df = test_df.apply(data.create_separator_model_input, axis=1)
        hidden_test_df = hidden_test_df.apply(data.create_separator_model_input, axis=1)

    elif args.input_type == 'diverse-tup-3':

        train_comet_dict, val_comet_dict, test_full_comet_dict, test_annotated_comet_dict = data.load_comet_relations(args.knowledge_folder)

        k=3
        diversity=True

        train_df = train_df.apply(lambda row: data.create_tup_model_input_with_k_comet_relations(row, train_comet_dict, k, diversity), axis=1)
        dev_df = dev_df.apply(lambda row: data.create_tup_model_input_with_k_comet_relations(row, val_comet_dict, k, diversity), axis=1)

        test_df = test_df.apply(lambda row: data.create_tup_model_input_with_k_comet_relations(row, test_full_comet_dict, k, diversity), axis=1)
        hidden_test_df = hidden_test_df.apply(lambda row: data.create_tup_model_input_with_k_comet_relations(row, test_annotated_comet_dict, k, diversity), axis=1)

        tokenizer.add_tokens(['<info>', '</info>'])

    elif args.input_type == 'diverse-tupsep-3':

        train_comet_dict, val_comet_dict, test_full_comet_dict, test_annotated_comet_dict = data.load_comet_relations(args.knowledge_folder)

        k=3
        diversity=True

        train_df = train_df.apply(lambda row: data.create_tupsep_model_input_with_k_comet_relations(row, train_comet_dict, k, diversity), axis=1)
        dev_df = dev_df.apply(lambda row: data.create_tupsep_model_input_with_k_comet_relations(row, val_comet_dict, k, diversity), axis=1)

        test_df = test_df.apply(lambda row: data.create_tupsep_model_input_with_k_comet_relations(row, test_full_comet_dict, k, diversity), axis=1)
        hidden_test_df = hidden_test_df.apply(lambda row: data.create_tupsep_model_input_with_k_comet_relations(row, test_annotated_comet_dict, k, diversity), axis=1)

        tokenizer.add_tokens(['<info>', '</info>'])

    elif args.input_type == 'diverse-verbalized-3':

        train_comet_dict, val_comet_dict, test_full_comet_dict, test_annotated_comet_dict = data.load_comet_relations(args.knowledge_folder)

        k=3
        diversity=True

        train_df = train_df.apply(lambda row: data.create_verbalized_model_input_with_k_comet_relations(row, train_comet_dict, k, diversity), axis=1)
        dev_df = dev_df.apply(lambda row: data.create_verbalized_model_input_with_k_comet_relations(row, val_comet_dict, k, diversity), axis=1)

        test_df = test_df.apply(lambda row: data.create_verbalized_model_input_with_k_comet_relations(row, test_full_comet_dict, k, diversity), axis=1)
        hidden_test_df = hidden_test_df.apply(lambda row: data.create_verbalized_model_input_with_k_comet_relations(row, test_annotated_comet_dict, k, diversity), axis=1)

    elif args.input_type == 'diverse-verbalized-1':

        train_comet_dict, val_comet_dict, test_full_comet_dict, test_annotated_comet_dict = data.load_comet_relations(args.knowledge_folder)

        k=1
        diversity=True

        train_df = train_df.apply(lambda row: data.create_verbalized_model_input_with_k_comet_relations(row, train_comet_dict, k, diversity), axis=1)
        dev_df = dev_df.apply(lambda row: data.create_verbalized_model_input_with_k_comet_relations(row, val_comet_dict, k, diversity), axis=1)

        test_df = test_df.apply(lambda row: data.create_verbalized_model_input_with_k_comet_relations(row, test_full_comet_dict, k, diversity), axis=1)
        hidden_test_df = hidden_test_df.apply(lambda row: data.create_verbalized_model_input_with_k_comet_relations(row, test_annotated_comet_dict, k, diversity), axis=1)

    elif args.input_type == 'diverse-verbalized-5':

        train_comet_dict, val_comet_dict, test_full_comet_dict, test_annotated_comet_dict = data.load_comet_relations(args.knowledge_folder)

        k=5
        diversity=True

        train_df = train_df.apply(lambda row: data.create_verbalized_model_input_with_k_comet_relations(row, train_comet_dict, k, diversity), axis=1)
        dev_df = dev_df.apply(lambda row: data.create_verbalized_model_input_with_k_comet_relations(row, val_comet_dict, k, diversity), axis=1)

        test_df = test_df.apply(lambda row: data.create_verbalized_model_input_with_k_comet_relations(row, test_full_comet_dict, k, diversity), axis=1)
        hidden_test_df = hidden_test_df.apply(lambda row: data.create_verbalized_model_input_with_k_comet_relations(row, test_annotated_comet_dict, k, diversity), axis=1)

    elif args.input_type == 'original-verbalized-3':

        train_comet_dict, val_comet_dict, test_full_comet_dict, test_annotated_comet_dict = data.load_comet_relations(args.knowledge_folder)

        k=3
        diversity=False

        train_df = train_df.apply(lambda row: data.create_verbalized_model_input_with_k_comet_relations(row, train_comet_dict, k, diversity), axis=1)
        dev_df = dev_df.apply(lambda row: data.create_verbalized_model_input_with_k_comet_relations(row, val_comet_dict, k, diversity), axis=1)

        test_df = test_df.apply(lambda row: data.create_verbalized_model_input_with_k_comet_relations(row, test_full_comet_dict, k, diversity), axis=1)
        hidden_test_df = hidden_test_df.apply(lambda row: data.create_verbalized_model_input_with_k_comet_relations(row, test_annotated_comet_dict, k, diversity), axis=1)

    elif args.input_type == 'reranked-verbalized-3':

        train_comet_dict, val_comet_dict, test_full_comet_dict, test_annotated_comet_dict = data.load_comet_relations(args.knowledge_folder)

        k=3
        diversity=False

        train_df = train_df.apply(lambda row: data.create_verbalized_model_input_with_k_comet_relations(row, train_comet_dict, k, diversity), axis=1)
        dev_df = dev_df.apply(lambda row: data.create_verbalized_model_input_with_k_comet_relations(row, val_comet_dict, k, diversity), axis=1)

        test_df = test_df.apply(lambda row: data.create_verbalized_model_input_with_k_comet_relations(row, test_full_comet_dict, k, diversity), axis=1)
        hidden_test_df = hidden_test_df.apply(lambda row: data.create_verbalized_model_input_with_k_comet_relations(row, test_annotated_comet_dict, k, diversity), axis=1)

    if args.mode == 'train':
        os.system(f'mkdir -p {args.model_dir}')

        create_model_config(args)

        model = T5ForConditionalGeneration.from_pretrained(args.model_arch, return_dict=True)

        model.resize_token_embeddings(len(tokenizer))
        logging.info('Loaded model and tokenizer')

        train_inputs = list(train_df['model_input'])
        dev_inputs = list(dev_df['model_input'])

        train_targets = list(train_df['answer'])
        dev_targets = list(dev_df['answer'])

        assert len(train_inputs) == len(train_targets)
        assert len(dev_inputs) == len(dev_targets)

        log_str = f'Experiment training data loaded of size {train_df.shape[0]} and dev data of size {dev_df.shape[0]}'
        logging.info(log_str)

        train_df.to_json(os.path.join(args.model_dir, 'train.json'), orient='records', indent=4)
        dev_df.to_json(os.path.join(args.model_dir, 'val.json'), orient='records', indent=4)

        finetune(train_df, dev_df, model, tokenizer, device, args)

    elif args.mode == 'test':

        model = torch.load(args.model_dir)

        if args.hidden_test:
            test_df = hidden_test_df

        model.resize_token_embeddings(len(tokenizer))
  
        logging.info('Loaded model and tokenizer')

        log_str = f'Loaded test data of size {test_df.shape[0]}'
        logging.info(log_str)

        predictions, targets = finetuned_model_inference(test_df, model, tokenizer, device, args)
        logging.info('Done with inference; writing to file...')

        out_fp = open(args.test_output_file, 'w+')
        fieldnames = ['meta', 'narrative', 'question', 'gold_answer', 'predicted_answer', 'is_ques_answerable']
        writer = csv.DictWriter(out_fp, fieldnames, lineterminator='\n', quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for idx, row in test_df.iterrows():
            info = {}
            question = row['question']
            context = row['narrative']
            info['meta'] = row['question_meta']
            info['narrative'] = context
            info['question'] = question
            info['gold_answer'] = targets[idx]
            info['is_ques_answerable'] = row['is_ques_answerable']
            info['predicted_answer'] = predictions[idx]
            writer.writerow(info)
        out_fp.close()
        logging.info('Done')

if __name__ == '__main__':
    args = parse_args()
    main(args)
