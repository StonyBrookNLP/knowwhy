"""
This file runs evaluation of model outputs using the cache of human judgments available
"""

import argparse
import sys
import pickle
import pandas as pd
import json
import re

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions-file-path', type=str, required=True, help='Predictions file path')
    parser.add_argument('--answers-file-path', type=str, required=True, help='Cache file')
    parser.add_argument('--output-file-path', type=str, required=True, help='Name of JSON metrics file that should be produced')
    args, _ = parser.parse_known_args()
    return args

def remove_punctuation(text):
    result = re.sub(r'[^\w\s]', '', text)
    return result

def retrieve_from_cache(cache, question, answer, narrative):

    """
    Use question, answer and narrative to retrieve all associated values
    Return failure if key not found
    """

    key = (question.lower(), answer.lower(), narrative.lower())
    try:
        return cache[key]
    except:
        return {'message': 'Key not found'}

def overall_avg_likert(df, cache):

    likerts = []
    for idx, row in df.iterrows():
        question = row['question']
        answer = remove_punctuation(row['predicted_answer'])
        narrative = row['narrative']
        key = (question, answer, narrative)
        try:
            info = retrieve_from_cache(cache, key[0], key[1], key[2])
            likertscores = info['val_annotations']
            likert = sum(likertscores) / len(likertscores)
            likerts.append(likert)
        except:
            pass

    return round(sum(likerts) / len(likerts), 2), round(len(likerts)*100 / df.shape[0], 2)

def overall_avg_binary_likert(df, cache):
    likerts = []
    for idx, row in df.iterrows():
        question = row['question']
        answer = remove_punctuation(row['predicted_answer'])
        narrative = row['narrative']
        key = (question, answer, narrative)
        try:
            info = retrieve_from_cache(cache, key[0], key[1], key[2])
            likertscores = info['val_annotations']
            binary_likertscores = [0 if x < 1 else 1 for x in likertscores]
            likert = sum(binary_likertscores) / len(binary_likertscores)
            likerts.append(likert)
        except:
            pass

    return round(sum(likerts) / len(likerts), 2), round(len(likerts)*100 / df.shape[0], 2)

def main(args):

    predictions_df = pd.read_csv(args.predictions_file_path)

    with open(args.answers_file_path, 'rb') as fp:
        cache = pickle.load(fp)

    avg_likert, coverage = overall_avg_likert(predictions_df, cache)
    binary_likert, coverage = overall_avg_binary_likert(predictions_df, cache)

    impl_predictions_df = predictions_df[predictions_df['is_ques_answerable'] == 'Not Answerable']

    impl_avg_likert, impl_coverage = overall_avg_likert(impl_predictions_df, cache)
    impl_binary_likert, impl_coverage = overall_avg_binary_likert(impl_predictions_df, cache)

    results = {'Avg Likert': avg_likert, 'Binary Accuracy': binary_likert, 'Coverage': coverage, 'Avg Likert (IMPL)': impl_avg_likert, 'Binary Accuracy (IMPL)': impl_binary_likert, 'Binary Coverage (IMPL)': impl_coverage}

    with open(args.output_file_path, 'w+') as fp:
        json.dump(results, fp)

if __name__=='__main__':
    args = parse_args()
    main(args)
