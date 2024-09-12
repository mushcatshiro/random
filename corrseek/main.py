import logging
import os

import pandas as pd

from bootstrap import *
from utils import DataGenerator


def main(
    fname: str,
    preprocess,
    datecol,
    ycol,
    groupcol
):
    # TODO 
    # to remove all except config fname from main fn args
    if not os.path.exists(fname):
        raise FileNotFoundError(
            f"file {fname} does not exists, "
            "please ensure correct directory is given or "
            "provide absolute file directory"
        )
    # ext = fname.split(".")[-1]
    # if ext == "csv":
    #     df = pd.read_csv(fname, index_col=0)
    # else:
    #     raise Exception(
    #         f"not supporting file ext: {ext}"
    #     )
    
    # X_cols = [
    #     x for x in df.columns.tolist() if x not in [ycol, datecol]
    # ]

    # if preprocess != 0:
    #     print("scaling")
    #     X = scale(df[X_cols], preprocess)
    # else:
    #     X = df[X_cols]
    # y = df[ycol]

    # m = LinearModel(threshold=0.007)
    # m.run(X, y)
    # print(m.report)
    d = DataGenerator(fname, ycol, datecol, groupcol, "csv")
    # m = LinearModel(topn=1)
    # for col in d.cols:
    #     for group in d.groups:
    #         y, X, _ = d.generate(col, group)
    #         m.run(X, y)
    #         break
    #     break


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filename",
        "-f",
        required=True,
        help=\
            "target filename"
    )
    parser.add_argument(
        "--preprocess",
        "-p",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help=\
            "0 no preprocessing; "\
            "1 normalize data to range 0~1; "\
            "2 normalize data to range -1~1"
    )
    parser.add_argument(
        "--datecolname",
        "-d",
        required=True,
        help=\
            "date column name"
    )
    parser.add_argument(
        "--ycolname",
        "-y",
        required=True,
        help=\
            "y column name "\
            "no support for multiple y yet "\
            "also non y/non date col are assumed to be X"
    )
    parser.add_argument(
        "--groupcol",
        "-g",
        required=True,
        help=\
            "group column values expect to be categorical data"
    )

    args = parser.parse_args()
    main(
        fname=args.filename,
        preprocess=args.preprocess,
        datecol=args.datecolname,
        ycol=args.ycolname,
        groupcol=args.groupcol
    )