# utils/data_utils.py

import json
import pandas as pd


def load_results(file_path):
    """
    Load JSONL file and parse into a DataFrame.
    """
    results = [json.loads(line) for line in open(file_path)]
    df = pd.DataFrame(results).explode(['generated_text', 'finish_reason'])
    df['ID'] = df['problem'].apply(lambda x: x['ID'])
    df['UNQ_ID'] = range(len(df))
    return df


def add_clinical_diag(row):
    """
    Adds clinical diagnosis label based on columns.
    """
    row['ID'] = row['NACCID']
    if (row['NACCALZP'] in [1, 2]) and (row['NACCLBDP'] in [1, 2]) and (
        row['FTLDMOIF'] in [1, 2] or row['FTLDNOIF'] in [1, 2] or row['FTDIF'] in [1, 2]):
        row["prediction"] = 'G'
    elif (row['NACCLBDP'] in [1, 2]) and (
        row['FTLDMOIF'] in [1, 2] or row['FTLDNOIF'] in [1, 2] or row['FTDIF'] in [1, 2]):
        row["prediction"] = 'F'
    elif (row['NACCALZP'] in [1, 2]) and (
        row['FTLDMOIF'] in [1, 2] or row['FTLDNOIF'] in [1, 2] or row['FTDIF'] in [1, 2]):
        row["prediction"] = 'E'
    elif (row['NACCALZP'] in [1, 2]) and (row['NACCLBDP'] in [1, 2]):
        row["prediction"] = 'D'
    elif row['FTLDMOIF'] in [1, 2] or row['FTLDNOIF'] in [1, 2] or row['FTDIF'] in [1, 2]:
        row["prediction"] = 'C'
    elif row['NACCLBDP'] in [1, 2]:
        row["prediction"] = 'B'
    elif row['NACCALZP'] in [1, 2]:
        row["prediction"] = 'A'
    else:
        row["prediction"] = 'H'
    return row


def prepare_test_data(data_cfg, ids):
    """
    Prepares the clinician label dataset.
    """
    test_data = pd.read_csv(data_cfg['test_data_path'])
    test_data_summary = pd.read_csv(data_cfg['test_data_summary_path'])

    test_data_subset = test_data[test_data['NACCID'].isin(ids)].reset_index(drop=True)
    test_data_summary_subset = test_data_summary[test_data_summary['ID'].isin(ids)].reset_index(drop=True)

    test_data_subset = test_data_subset[[
        'NACCID', 'NACCALZP', 'NACCLBDP', 'FTLDMOIF', 'FTLDNOIF', 'FTDIF'
    ]].apply(add_clinical_diag, axis=1)

    test_data_subset.drop(['NACCID'], axis=1, inplace=True)
    merged = test_data_summary_subset.merge(test_data_subset, on='ID')
    return merged
