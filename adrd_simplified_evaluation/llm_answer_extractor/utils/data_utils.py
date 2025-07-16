# utils/data_utils.py

import json
import pandas as pd
import ast
import re

def load_results(file_path):
    """
    Load JSONL file and parse into a DataFrame.
    """
    results = [json.loads(line) for line in open(file_path)]
    df = pd.DataFrame(results).explode(['generated_text', 'finish_reason'])
    if isinstance(df.iloc[0]['problem'], str):
        df['problem'] = df['problem'].apply(lambda x: ast.literal_eval(x))
    df['ID'] = df['problem'].apply(lambda x: x['ID'])
    # df['Q_TYPE'] = df['problem'].apply(lambda x: x['Q_TYPE'])
    df['UNQ_ID'] = range(len(df))
    return df


# def add_clinical_diag(row):
#     """
#     Adds clinical diagnosis label based on columns.
#     """
#     row['ID'] = row['NACCID']
#     if (row['NACCALZP'] in [1, 2]) and (row['NACCLBDP'] in [1, 2]) and (
#         row['FTLDMOIF'] in [1, 2] or row['FTLDNOIF'] in [1, 2] or row['FTDIF'] in [1, 2]):
#         row["prediction"] = 'G'
#     elif (row['NACCLBDP'] in [1, 2]) and (
#         row['FTLDMOIF'] in [1, 2] or row['FTLDNOIF'] in [1, 2] or row['FTDIF'] in [1, 2]):
#         row["prediction"] = 'F'
#     elif (row['NACCALZP'] in [1, 2]) and (
#         row['FTLDMOIF'] in [1, 2] or row['FTLDNOIF'] in [1, 2] or row['FTDIF'] in [1, 2]):
#         row["prediction"] = 'E'
#     elif (row['NACCALZP'] in [1, 2]) and (row['NACCLBDP'] in [1, 2]):
#         row["prediction"] = 'D'
#     elif row['FTLDMOIF'] in [1, 2] or row['FTLDNOIF'] in [1, 2] or row['FTDIF'] in [1, 2]:
#         row["prediction"] = 'C'
#     elif row['NACCLBDP'] in [1, 2]:
#         row["prediction"] = 'B'
#     elif row['NACCALZP'] in [1, 2]:
#         row["prediction"] = 'A'
#     else:
#         row["prediction"] = 'H'
#     return row


def get_clinical_and_desc(row):
    """
    Adds clinical diagnosis label based on columns.
    """
    if (row['NACCALZP'] in [1, 2]) and (row['NACCLBDP'] in [1, 2]) and (
        row['FTLDMOIF'] in [1, 2] or row['FTLDNOIF'] in [1, 2] or row['FTDIF'] in [1, 2]):
        row["prediction_text"] = "Alzheimer's disease pathology (AD), Lewy body pathology (LBD) and Frontotemporal Lobar Degeneration with tau pathology or TDP-43 pathology (FTLD)"
    elif (row['NACCLBDP'] in [1, 2]) and (
        row['FTLDMOIF'] in [1, 2] or row['FTLDNOIF'] in [1, 2] or row['FTDIF'] in [1, 2]):
        row["prediction_text"] = "Lewy body pathology (LBD) and Frontotemporal Lobar Degeneration with tau pathology or TDP-43 pathology (FTLD)"
    elif (row['NACCALZP'] in [1, 2]) and (
        row['FTLDMOIF'] in [1, 2] or row['FTLDNOIF'] in [1, 2] or row['FTDIF'] in [1, 2]):
        row["prediction_text"] = "Alzheimer's disease pathology (AD) and Frontotemporal Lobar Degeneration with tau pathology or TDP-43 pathology (FTLD)"
    elif (row['NACCALZP'] in [1, 2]) and (row['NACCLBDP'] in [1, 2]):
        row["prediction_text"] = "Alzheimer's disease pathology (AD) and Lewy body pathology (LBD)"
    elif row['FTLDMOIF'] in [1, 2] or row['FTLDNOIF'] in [1, 2] or row['FTDIF'] in [1, 2]:
        row["prediction_text"] = "Frontotemporal Lobar Degeneration with tau pathology or TDP-43 pathology (FTLD) only"
    elif row['NACCLBDP'] in [1, 2]:
        row["prediction_text"] = "Lewy body pathology (LBD) only"
    elif row['NACCALZP'] in [1, 2]:
        row["prediction_text"] = "Alzheimer's disease pathology (AD) only"
    else:
        row["prediction_text"] = "No listed option is correct"
    return row

def add_clinical_diag(row):
    matches = re.findall(r'([A-Z])\. (.+)', row['options'])

    # Create dictionary: full text => letter
    options_dict = {option: letter for letter, option in matches}
    
    row['prediction'] = options_dict[row['prediction_text']]
    
    return row


def prepare_test_data(data_cfg):
    """
    Prepares the clinician label dataset.
    """
    test_data = pd.read_csv(data_cfg['test_data_path'])
    test_data_summary = pd.read_csv(data_cfg['test_data_summary_path'])

    test_data_subset = test_data[test_data['NACCID'].isin(test_data_summary['ID'])].reset_index(drop=True)
    # test_data_summary_subset = test_data_summary[test_data_summary['ID'].isin(ids)].reset_index(drop=True)

    # test_data_subset = test_data_subset[[
    #     'NACCID', 'NACCALZP', 'NACCLBDP', 'FTLDMOIF', 'FTLDNOIF', 'FTDIF'
    # ]].apply(add_clinical_diag, axis=1)
    
    test_data_subset['ID'] = test_data_subset['NACCID']
    test_data_subset.drop(['NACCID'], axis=1, inplace=True)
    
    common_columns = list(set(test_data_subset.columns).intersection(set(test_data_summary.columns)))
    merged = test_data_summary.merge(test_data_subset, on=common_columns)
    merged = merged[list(test_data_summary.columns) + [
        'NACCALZP', 'NACCLBDP', 'FTLDMOIF', 'FTLDNOIF', 'FTDIF'
    ]].apply(get_clinical_and_desc, axis=1)
    
    merged = merged.apply(add_clinical_diag, axis=1)
    
    return merged
