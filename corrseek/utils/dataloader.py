import os

import pandas as pd


class DataLoader:
    def __init__(self, full_dir) -> None:
        if not os.path.isfile(full_dir):
            raise ValueError(
                f"file does not exists {full_dir}"
            )
        self.df = pd.read_csv(full_dir)

    def generate(self, cond):
        pass