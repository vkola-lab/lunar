# pipeline/evaluator.py

import pandas as pd
import numpy as np


class Evaluator:
    def __init__(self, model_dfs, k=1):
        self.model_dfs = model_dfs
        self.k = k
        self.modified = {name: self._modify(df) for name, df in model_dfs.items()}
        self.n = len(self.modified['clinician'])

    def _modify(self, df):
        df_mod = df[['prediction', 'ground_truth', 'ID']].copy()
        df_mod['correctness'] = df_mod['prediction'] == df_mod['ground_truth']
        df_mod = df_mod.reset_index(names=['problem']).reset_index(drop=True)
        return df_mod

    def _pass_at_k(self, df, n, k):
        cs = df.groupby('problem').sum('correctness')
        vals = []
        for _, row in cs.iterrows():
            c = row['correctness']
            if n - c < k:
                vals.append(1.0)
            else:
                vals.append(1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))
        return np.mean(vals)

    def _cons_at_k(self, df):
        return (
            df.groupby('problem')['prediction']
            .apply(lambda x: x.mode()[0]) ==
            df[['problem', 'ground_truth']].drop_duplicates('problem')['ground_truth'].reset_index(drop=True)
        ).sum() / self.n

    def evaluate(self):
        final_dict = {'metric': ['pass@1', 'cons@k']}
        for name, df in self.modified.items():
            p = len(df) // self.n
            final_dict[name] = [self._pass_at_k(df, p, self.k), self._cons_at_k(df)]
        return pd.DataFrame(final_dict)
