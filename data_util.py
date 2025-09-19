import torch
import numpy as np
import pandas as pd

from typing import Iterable
from typing import Optional
from typing import Union


def calculate_entropy(prompt: str):
    entropy = 0
    letters = "".join(set(prompt))

    for i in letters:
        x = prompt.count(i)
        y = x = x / len(prompt)
        x = np.log2(x)
        x = x * y
        entropy = entropy + x
    return -entropy


def pretty_print_sim_idx(
    names, S, decimals=3, hide_diag=True, start=0, pad=2, show_map=True, maxlen=60
):
    """
    names  : list of categories full names (np. ["passage: ...", ...])
    S      : similiarity matrix (NxN) - torch.Tensor or np.ndarray
    decimals: liczba miejsc po przecinku
    hide_diag: replace diagonal by '—' for easier reading
    start  : indexing starting index
    pad    : number of digits in the index (e.g. 2 -> 00, 01, 02)
    show_map: whether to print index -> name mapping
    maxlen : max. row length in mapping (print only)
    """
    # 1) clean out category names
    clean = [n.removeprefix("passage: ").strip() for n in names]

    # 2) prepare the matrix for numpy
    if isinstance(S, torch.Tensor):
        S = S.detach().cpu().numpy()
    S = np.asarray(S)

    # 3) labels = index only
    labels = [f"{i + start:0{pad}d}" for i in range(len(clean))]

    # 4) DataFrame
    df = pd.DataFrame(S, index=labels, columns=labels)

    # 5) hide diagonal (better readability)
    if hide_diag:
        np.fill_diagonal(df.values, np.nan)

    # 6) print out the table
    print(
        df.to_string(float_format=lambda x: f"{x:.{decimals}f}" if pd.notna(x) else "—")
    )

    # 7) optional: index → full name mapping
    if show_map:
        print("\nMapowanie indeksów:")
        for idx, name in zip(labels, clean):
            t = name if len(name) <= maxlen else name[: maxlen - 1] + "…"
            print(f"  {idx}  {t}")


def extract_column_from_csv(
    filename: str, column_idx: int, data_type: type
) -> pd.DataFrame:
    """
    extracting a single @filename.csv column at index @column_idx as @data_type
    """
    df = pd.read_csv(
        filename, sep=";", header=0, usecols=[column_idx], skip_blank_lines=False
    )
    results = df.iloc[:, 0].astype(data_type)
    return results


def extract_columns(
    filename: str,
    columns: Optional[Union[Iterable[int], slice, range]] = None,
    dtype: Optional[Union[type, dict]] = None,
    sep: str = ";",
    header: int = 0,
    skip_blank_lines: bool = False,
    return_numpy: bool = False,
):
    """
    Reads selected columns from CSV file.
    - columns: index list/set, slice (e.g. slice(3, None, 2)) or range.
               if None -> all columns.
    - dtype: single type (e.g. float) or dictionary {column: type}.
             works like DataFrame.astype(...).
    - return_numpy: True -> returns (np.ndarray), False -> pd.DataFrame.
    """
    # if columns is a slice with no .stop set,
    # we need to know the number of columns
    if isinstance(columns, slice) and columns.stop is None:
        header_only = pd.read_csv(filename, sep=sep, header=header, nrows=0)
        ncols = header_only.shape[1]
        usecols = list(range(columns.start or 0, ncols, columns.step or 1))
    elif isinstance(columns, (range, list, tuple, set)):
        usecols = list(columns)  # pandas requires list/tuple
    else:
        usecols = columns  # None or full slice with set .stop

    df = pd.read_csv(
        filename,
        sep=sep,
        header=header,
        usecols=usecols,
        skip_blank_lines=skip_blank_lines,
    )

    if dtype is not None:
        df = df.astype(dtype)

    return df.to_numpy() if return_numpy else df
