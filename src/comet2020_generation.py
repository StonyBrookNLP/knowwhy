"""
This file contains code to run generation with COMET2020.
It has the default generation example from the AI2 COMET2020 repository as well as functions written by me for ease of use.

Most of these functions are copied from:
1. https://github.com/allenai/comet-atomic-2020/blob/f5fe091810539707dfbc962f6a6c04f557575d5b/models/comet_atomic2020_bart/utils.py
2. https://github.com/allenai/comet-atomic-2020/blob/f5fe091810539707dfbc962f6a6c04f557575d5b/models/comet_atomic2020_bart/generation_example.py
"""

import torch
import random
import numpy as np
import pandas as pd
import os
import json
import argparse

from collections import defaultdict
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

np.random.seed(1234)
torch.manual_seed(1234)
random.seed(1234)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', type=str, help='Input JSON file path')
    parser.add_argument('--model-path', type=str, default='../models/comet-atomic_2020_BART', help='Directory with saved COMET2020 model to be used')
    parser.add_argument('--output-file', type=str, help='Output JSON file path')
    parser.add_argument('--num-rel-per-type', type=int, default=3, help='Number of relations to generate per type')

    args, _ = parser.parse_known_args()
    return args

def trim_batch(
    input_ids, pad_token_id, attention_mask=None,
):
    """Remove columns that are populated exclusively by pad_token_id"""
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
    if attention_mask is None:
        return input_ids[:, keep_column_mask]
    else:
        return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask])

def use_task_specific_params(model, task):
    """Update config with summarization specific params."""
    task_specific_params = model.config.task_specific_params

    if task_specific_params is not None:
        pars = task_specific_params.get(task, {})
        model.config.update(pars)

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]

class Comet:
    def __init__(self, model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        task = "summarization"
        use_task_specific_params(self.model, task)
        self.batch_size = 1
        self.decoder_start_token_id = None

    def generate(
            self, 
            queries,
            decode_method="beam", 
            num_generate=5, 
            ):

        with torch.no_grad():
            examples = queries

            decs, scores = [], []
            for batch in list(chunks(examples, self.batch_size)):

                batch = self.tokenizer(batch, return_tensors="pt", truncation=True, padding="max_length").to(self.device)
                input_ids, attention_mask = trim_batch(**batch, pad_token_id=self.tokenizer.pad_token_id)

                summaries = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_start_token_id=self.decoder_start_token_id,
                    num_beams=num_generate,
                    num_return_sequences=num_generate,
                    output_scores=True,
                    return_dict_in_generate=True
                    )

                dec = self.tokenizer.batch_decode(summaries['sequences'], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                decs.append(dec)
                scores.append(summaries['sequences_scores'].tolist())

            return decs, scores

def run_generation(comet, prompt, desired_relations, decode_method="beam", num_generate=3):

    """
    COMET2020 will generate outputs for a list of desired relations for an event chunk
    For each relation, it will generate num_generate number of outputs (can include 'none')
    It will use decode_method for decoding
    """

    queries = []
    for rel in desired_relations:
        query = "{} {} [GEN]".format(prompt, rel)
        queries.append(query)
    results, scores = comet.generate(queries, decode_method, num_generate)
    return results, scores

def clean_raw_comet(data, out_fname):

    """
    Take raw comet dict as input and create a new file with comet data
    For each meta, there will a list of dictionaries; dict keys are score, type and phrase
    """

    formatted_data = {}
    for meta, comet_info in data.items():
        comet_data = []
        for rel_type, rel_info in comet_info.items():
            for i in range(0, len(rel_info), 2):
                comet_data.append({'rel_type': rel_type, 'rel_phrase': rel_info[0], 'rel_score': rel_info[1]})
        formatted_data[meta] = comet_data
    with open(out_fname, 'w+') as fp:
        json.dump(formatted_data, fp, indent=4)

def enrich_tellmewhy(df, args):

    """
    Use TellMeWhy file in Pandas dataframe format and extract specific types of COMET relations for each question
    Generate 3 relations of each type
    """

    model_path = args.model_path
    comet = Comet(model_path)
    comet.model.zero_grad()
    print(f'Loaded COMET2020 model from {model_path}')

    desired_relations = [
        "Causes",
        "CausesDesire",
        "DesireOf",
        "Desires",
        "HasFirstSubevent",
        "HasLastSubevent",
        "HasPrerequisite",
        "HasSubEvent",
        "HinderedBy",
        "MotivatedByGoal",
        "oEffect",
        "oReact",
        "oWant",
        "xEffect",
        "xIntent",
        "xNeed",
        "xReact",
        "xReason",
        "xWant",
    ]

    comet_info = {}
    df = df.drop_duplicates(subset='question_meta', ignore_index=True)
    for idx, row in tqdm(df.iterrows()):
        meta = row['question_meta']
        narrative_as_list = row['original_narrative_form']
        original_sentence_for_question = row['original_sentence_for_question']
        original_sentence_idx = narrative_as_list.index(original_sentence_for_question)

        sentence = narrative_as_list[original_sentence_idx]

        if sentence[-1] == '.' or sentence[-1] == '!':
            sentence = sentence[:-1]

        comet_outputs, comet_scores = run_generation(comet, sentence, desired_relations, 'beam', args.num_rel_per_type)
        assert len(comet_outputs) == len(desired_relations)

        info = defaultdict(list)
        for relation, comet_output, comet_score in zip(desired_relations, comet_outputs, comet_scores):
            for output, score in zip(comet_output, comet_score):
                info[relation].extend((output, score))

        comet_info[meta] = info

    clean_raw_comet(comet_info, args.output_file)

def main(args):

    df = pd.read_json(args.input_file)
    enrich_tellmewhy(df, args)

if __name__ == '__main__':
    args = parse_args()
    main(args)
