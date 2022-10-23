import logging
import os

import pandas as pd

from bootstrap import *


def main(
    fname: str,
    preprocess,
    multicolinear,
    datecol,
    ycol
):
    if not os.path.exists(fname):
        raise FileNotFoundError(
            f"file {fname} does not exists, "
            "please ensure correct directory is given or "
            "provide absolute file directory"
        )
    ext = fname.split(".")[-1]
    if ext == "csv":
        df = pd.read_csv(fname, index_col=0)
    else:
        raise Exception(
            f"not supporting file ext: {ext}"
        )
    
    X_cols = [
        x for x in df.columns.tolist() if x not in [ycol, datecol]
    ]

    if preprocess != 0:
        print("scaling")
        X = scale(df[X_cols], preprocess)
    else:
        X = df[X_cols]
    y = df[ycol]

    m = LinearModel(threshold=0.007)
    m.run(X, y)
    print(m.report)




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
        "--multicolinearity",
        "-m",
        type=int,
        default=0,
        choices=[0, 1],
        help=\
            "0 skip multicolinearity; "\
            "1 run multicolinearity"
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

    args = parser.parse_args()
    main(
        fname=args.filename,
        preprocess=args.preprocess,
        multicolinear=args.multicolinearity,
        datecol=args.datecolname,
        ycol=args.ycolname
    )