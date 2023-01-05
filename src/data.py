# This file contains functions to load datasets and manipulate it to correct model input format

import pandas as pd
import os
import json
import argparse

revised_template_map = {
    "causes": "causes",
    "causesdesire": "makes someone want",
    "desireof": "is a desire of",
    "desires": "desires",
    "hasfirstsubevent": "begins with",
    "haslastsubevent": "ends with",
    "hasprerequisite": "to do this, one requires",
    "hassubevent": "includes",
    "hinderedby": "can be hindered by",
    "motivatedbygoal": "is a step towards accomplishing",
    "oeffect": "as a result, they will",
    "oreact": "as a result, they feel",
    "owant": "as a result, they want",
    "xeffect": "as a result, she will",
    "xintent": "because she wanted",
    "xneed": "but before, she needed",
    "xreact": "as a result, she feels",
    "xreason": "because",
    "xwant": "as a result, she wants",
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--comet-data-folder', type=str, help='Folder with relevant COMET relations in specific format')
    args, _ = parser.parse_known_args()
    return args

def get_verbalized_comet_relation(comet_prompt, rel_type, rel_phrase):

    """
    Use the original sentence for question, the relation phrase and its type 
    to create a fluent COMET inference rather than the phrases it spits out.
    """

    rel_phrase = rel_phrase.lower()
    px_pos = rel_phrase.find('personx')
    py_pos = rel_phrase.find('persony')
    if px_pos == -1:
        rel_phrase = rel_phrase.replace('persony', 'she', 1)
        rel_phrase = rel_phrase.replace('persony', 'her')
    elif py_pos == -1:
        rel_phrase = rel_phrase.replace('personx', 'she', 1)
        rel_phrase = rel_phrase.replace('personx', 'her')
    elif px_pos < py_pos:
        rel_phrase = rel_phrase.replace('personx', 'she')
        rel_phrase = rel_phrase.replace('persony', 'her')
    elif py_pos < px_pos:
        rel_phrase = rel_phrase.replace('persony', 'she')
        rel_phrase = rel_phrase.replace('personx', 'her')

    return f'{comet_prompt} {revised_template_map[rel_type.lower()]}, {rel_phrase}'

def get_topk_comet_relations(comet_relations, k, diversity=False):

    """
    From list of COMET relations, retrieve top k relations, where k is specified and also account for diversity (no two relations in returned ones can be of same type)
    """

    count_k = 0
    all_relation_types = list(revised_template_map.keys())

    relations = []
    for relation in comet_relations:
        rel_phrase = relation['rel_phrase'].lstrip().rstrip()
        if rel_phrase.lower() == 'none':
            continue
        if diversity:
            relations.append(relation)
        else:
            rel_type = relation['rel_type']
            if rel_type in all_relation_types:
                relations.append(relation)
                all_relation_types.remove(rel_type)
            else:
                continue
        count_k += 1
        if count_k == k:
            break

    return relations

def create_normal_model_input(row):

    """
    TellMeWhy paper baseline
    question: Q context: C
    """

    narrative = row['narrative']
    question = row['question']
    model_input = 'question: ' + question + ' context: ' + narrative
    row['model_input'] = model_input
    return row

def create_separator_model_input(row):

    """
    question: Q \\n context: C
    """

    narrative = row['narrative']
    question = row['question']
    model_input = 'question: ' + question + '\\n context: ' + narrative
    row['model_input'] = model_input

    return row

def create_tup_model_input_with_k_comet_relations(row, comet_dict, k, diversity=False):

    """
    Add k relations in the below format
    question: Q context: C <info> relation: {rel_type} phrase: {rel_phrase} </info>
    """

    meta = row['question_meta']
    comet_relations = comet_dict[meta]
    question = row['question']
    story = row['narrative']

    model_input = f'question: {question} context: {story}'

    relations = get_topk_comet_relations(comet_dict[meta], k, diversity)

    for relation in relations:
        rel_type = relation['rel_type']
        rel_phrase = relation['rel_phrase'].lstrip().rstrip()
        model_input = f"{model_input} <info> relation: {relation['rel_type']} phrase: {relation['rel_phrase']} </info>"

    row['model_input'] = model_input

    return row

def create_tupsep_model_input_with_k_comet_relations(row, comet_dict, k, diversity=False):

    """
    Add k relations in the below format
    question: Q \\n context: C \\n <info> relation: {rel_type} phrase: {rel_phrase} </info> \\n
    """

    meta = row['question_meta']
    comet_relations = comet_dict[meta]
    question = row['question']
    story = row['narrative']

    model_input = f'question: {question} \\n context: {story} \\n'

    relations = get_topk_comet_relations(comet_dict[meta], k, diversity)

    for relation in relations:
        rel_type = relation['rel_type']
        rel_phrase = relation['rel_phrase'].lstrip().rstrip()
        model_input = f"{model_input} <info> relation: {relation['rel_type']} phrase: {relation['rel_phrase']} </info> \\n"

    row['model_input'] = model_input

    return row

def create_verbalized_model_input_with_k_comet_relations(row, comet_dict, k, diversity=False):

    """
    Add k relations in the below format
    question: Q \\n context: C \\n verbalized_relation \\n
    """

    meta = row['question_meta']
    comet_relations = comet_dict[meta]
    question = row['question']
    story = row['narrative']
    original_sentence_for_question = row['original_sentence_for_question']

    model_input = f'question: {question} \\n context: {story} \\n'

    relations = get_topk_comet_relations(comet_dict[meta], k, diversity)

    for relation in relations:
        rel_type = relation['rel_type']
        rel_phrase = relation['rel_phrase'].lstrip().rstrip()
        verbalized_comet_relation = get_verbalized_comet_relation(original_sentence_for_question, rel_type, rel_phrase)
        model_input = f'{model_input} {verbalized_comet_relation} \\n'

    row['model_input'] = model_input

    return row

def load_comet_relations(data_folder):

    """
    Load differently ordered COMET relations for TellMeWhy
    File format: {meta: [{'rel_type': type_, 'rel_phrase': relation}, ...]}
    Each dictionary that describes a relation can also have 'rel_score' key
    """

    with open(os.path.join(data_folder, 'train.json'), 'r') as fp:
        train_relations = json.load(fp)
    with open(os.path.join(data_folder, 'val.json'), 'r') as fp:
        val_relations = json.load(fp)
    with open(os.path.join(data_folder, 'test_full.json'), 'r') as fp:
        test_full_relations = json.load(fp)
    with open(os.path.join(data_folder, 'test_annotated.json'), 'r') as fp:
        test_annotated_relations = json.load(fp)

    return train_relations, val_relations, test_full_relations, test_annotated_relations

def load_tellmewhy(data_folder):

    """
    Load TellMeWhy original data for local folder
    """

    train_df = pd.read_json(os.path.join(data_folder, 'train.json'))
    dev_df = pd.read_json(os.path.join(data_folder, 'val.json'))
    test_df = pd.read_json(os.path.join(data_folder, 'test_full.json'))
    hidden_test_df = pd.read_json(os.path.join(data_folder, 'test_annotated.json'))
    return train_df, dev_df, test_df, hidden_test_df

def main(args):
    t, v, tf, ta = load_comet_relations(args.comet_data_folder)

if __name__=='__main__':
    args = parse_args()
    main(args)