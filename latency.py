"""latency script & utils"""
import argparse
import json
import os
from pathlib import Path

import pandas as pd

RELIABLE_TIME = RELIABLE_CYCLES = ''


def get_suffix(filename: Path, pattern='', merge=True):
    """Extracts the suffix from the given filename & pattern.

    Parameters:
        - filename (Path): The input filename.
        - pattern (str, optional): The pattern to split the filename.
        - merge (bool, optional): If True, multiple segments after splitting
          are merged with underscores.

    Returns:
        str: The extracted suffix from the filename.
    """
    pattern, *suffix = str(filename.parent).split(pattern)
    if not isinstance(suffix, str):
        suffix = '_'.join(suffix)
    return suffix


def parse_automcaml_qnn_json(filename, args):
    df, data = parse_qnn(filename)
    if args.k <= 0:
        args.k = len(df)
    df_sorted, sum_cycles = sort_by_latency(df)

    global RELIABLE_TIME, RELIABLE_CYCLES
    if sum_cycles > 0:
        RELIABLE_TIME = '[⚠️UNRELIABLE⚠️] '
    else:
        RELIABLE_CYCLES = '[⚠️UNRELIABLE⚠️] '
    print(f'{RELIABLE_TIME}Latency (ms): {data["latency"]}')
    print(f'{RELIABLE_CYCLES}Sum cycles: {sum_cycles}')
    print(f'Number operators: {len(df)}')
    if sum_cycles > 0 or args.show_shape:
        print(f'Top-{args.k} most expensive')
        print(df_sorted.head(n=args.k).to_markdown())

    with open(args.output, 'w') as fid:
        fid.write(f'{RELIABLE_TIME}Latency (ms): {data["latency"]}\n')
        fid.write(f'{RELIABLE_CYCLES}Sum cycles: {sum_cycles}\n')
        fid.write(f'Number operators: {len(df)}\n')
        if sum_cycles > 0 or args.show_shape:
            fid.write(f'Top-{args.k} most expensive\n')
            fid.write(df_sorted.head(n=args.k).to_markdown())
    return df_sorted, data, sum_cycles


def parse_qnn(filename: Path) -> pd.DataFrame:
    with open(filename, 'r') as fid:
        data = json.load(fid)

    df = pd.DataFrame(data['layers'])
    df = df.transpose()
    return df, data


def pattern_finder(str_path_list):
    """Infers and returns a common prefix pattern from a list of string paths.

    Parameters:
        - str_path_list (List[str]): List of string paths to infer the common
            prefix pattern.

    Returns:
        str: The inferred common prefix pattern.
    """
    print('Inferring pattern...')
    pattern = os.path.commonprefix(str_path_list)
    pattern = pattern.split('_')[0] + '_'
    print(f'Pattern to trim: {pattern}')
    return pattern


def sort_by_latency(df: pd.DataFrame)-> pd.DataFrame:
    df_sorted = df.sort_values(by='latency', ascending=False)
    sum_cycles = df_sorted["latency"].sum()
    df_sorted['rel cycles'] = df_sorted["latency"] / sum_cycles
    return df_sorted, sum_cycles


def depreacted():
    import numpy as np
    n = np.array(n)
    latency = np.array(latency)
    eta = latency * (28 / n)
    ind_min = np.argmin(eta)
    ind_max = np.argmax(eta)
    print(f'Latency range: [{eta[ind_min]}, {eta[ind_max]}] msec')
    print(f'Min/Max blocks {n[ind_min]}, {n[ind_max]}')
    print(f'Avg latency: {np.mean(eta)}')


def main(args):
    pattern_guessed = False
    if args.pattern is None:
        pattern_guessed = True
        args.pattern = pattern_finder([i.parent for i in args.filename])

    exp_ids, results = set(), []
    for json_i in args.filename:
        if args.output is None:
            args.output = json_i.parent / f'report_ops.txt'
        _, data, _ = parse_automcaml_qnn_json(json_i, args)
        name = get_suffix(json_i, args.pattern)
        results.append((name, data['latency']))
        exp_ids.add(name)
        print(f'{name},{data["latency"]}')
    num_profiles = len(results)
    msg = ''
    if num_profiles >= 1:
        msg = '🎉 '
    print(f'{msg}Processed latency reports: {num_profiles}')
    if num_profiles > 1:
        unique_parsing = len(exp_ids) == num_profiles
        if not unique_parsing:
            msg = '[⚠️WARNING⚠️] Repeated Experiment IDs.'
            if pattern_guessed:
                msg += f' {pattern_guessed=}, consider providing a pattern.'
            else:
                msg += ' Review pattern & suffix guess functions'
            print(msg)
        results.sort()
        print('ProfileResultsExpId,latency(msec)')
        if unique_parsing or args.force:
            for name, latency in results:
                print(f'{name},{latency}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Parse QNN Profile JSON file")
    parser.add_argument('--filename', '-f', type=Path, nargs='+',
                        help='Path to the QNN JSON file',
                        default='checkpoints/model-it-ith/model.iters-100.qnn.int8.json')
    parser.add_argument('-k', type=int, default=-1,
                        help='Number of top most expensive operators to display')
    parser.add_argument('--output', '-o', type=Path, default=None)
    parser.add_argument('--show_shape', '-ss', action='store_true', default=False,
                        help='Print output shape per op')
    parser.add_argument('--pattern', '-p', type=str, default=None)
    parser.add_argument('--force', '-ff', action='store_true')

    args = parser.parse_args()
    if args.force:
        args.show_shape = True

    main(args)
