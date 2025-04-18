import argparse
import logging
import os
import re
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description='Parse SpeechBrain training logs to analyze and summarize results.',
                                     usage='%(prog)s train-log1.txt train-log2.txt ... train-logN.txt --epochs 110')
    parser.add_argument('log_files', nargs='+', help='Paths to one or more log text files. They must cover'
                                                     ' the entire training phase as well as be in order.')
    parser.add_argument('--epochs', type=int, required=True, help='Total number of epochs performed in the'
                                                                  ' training.')
    parser.add_argument('--df_outfile', default='training_res.csv', help='CSV file to save training results.')
    parser.add_argument('--plots_outdir', default='./img', help='Directory to save plots in.')

    args = parser.parse_args()

    # Gather data
    df = construct_df(args.log_files)
    validate_df(df, args.epochs)
    df = df.sort_values(by=['epoch'])
    df.to_csv(args.df_outfile, index=False)
    print('Validated training results df and saved it: ' + args.df_outfile)

    # Visualize results
    handle_plots(df, args.plots_outdir)
    print(f'Done.\nPlots were saved to {args.plots_outdir}.')


def construct_df(filenames: list[str]) -> pd.DataFrame:
    """
    Constructs a dataframe containing in-training results, from the supplied log files.
    :param filenames: SpeechBrain log files (complete and in order)
    :return: Dataframe containing data such as train and val loss, through epochs
    """

    records = []
    for file in filenames:
        with open(file, 'r') as f:
            for line in f:
                parsed = parse_line(line)
                if parsed:
                    records.append(parsed)

    return pd.DataFrame(records)


def parse_line(line: str) -> Optional[dict]:
    """
    Parse a single training log line.
    :param line: line to parse
    :return: a record of its data, as a dictionary; or None in case the line is irrelevant/mal-formatted
    """

    preamble = 'speechbrain.utils.train_logger - '  # separates relevant lines from irrelevant
    if not line.startswith(preamble):
        return None

    match = re.search(r'epoch: (?P<epoch>\d+), lr: (?P<lr>[-+e\d.]+), steps: (?P<steps>\d+), optimizer: \w+'
                      r' - train loss: (?P<train_loss>[-+e\d.]+) - valid loss: (?P<valid_loss>[-+e\d.]+), valid '
                      r'ACC: (?P<valid_acc>[-+e\d.]+)(?:, valid WER: (?P<valid_wer>[-+e\d.]+))?',
                      line[len(preamble):])
    if not match:
        return None  # line is not well-formatted
    else:
        return {  # convert fields to their respective numeric types
            'epoch': int(match['epoch']),
            'lr': float(match['lr']),
            'steps': int(match['steps']),
            'train_loss': float(match['train_loss']),
            'val_loss': float(match['valid_loss']),
            'val_acc': float(match['valid_acc']),
            'val_wer': float(match['valid_wer']) if match['valid_wer'] is not None else None
        }


def validate_df(df: pd.DataFrame, expected_epochs: int) -> None:
    """
    Validates that all expected epochs are present and no cell is missing that's not supposed to.

    It gets the dataframe as well as the number of epochs that were performed (so it could check all are
    recorded). It prints warning if there's something suspicious, and only if error is fatal an exception is raised.
    """

    missing_epochs = set(range(1, expected_epochs + 1)) - set(df['epoch'])
    if missing_epochs:
        logging.warning(f'Missing epochs in logs: {sorted(missing_epochs)}')

    cols = ['epoch', 'lr', 'steps', 'train_loss', 'val_loss', 'val_acc']
    for col in cols:
        if col not in df.columns or df[col].isnull().any():
            raise ValueError('Fatal - this column is missing from some rows: ' + col)

    # Only val_wer column isn't mandatory everywhere; but give a warning if it does not appear anywhere.
    if df['val_wer'].isnull().all():
        logging.warning('\'valid_wer\' column has all of its entries missing; the corresponding plot will hence be'
                        'skipped.')

def handle_plots(df: pd.DataFrame, outdir: str) -> None:
    """
    Saves visualization plots for the performed training.
    Three plots are made:

    :param df: dataframe of in-training results, sorted
    :param outdir: where to save the plots
    """

    os.makedirs(outdir, exist_ok=True)

    epochs = df['epoch']

    # Training and validation loss
    plt.figure()
    plt.plot(epochs, df['train_loss'], label='Train Loss')
    plt.plot(epochs, df['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss through the Epochs')
    plt.legend()
    save_plot('loss', outdir)

    #
    wer_data = df.dropna(subset=['val_wer'])
    if not wer_data.empty:
        plt.figure()
        plt.plot(wer_data['epoch'], wer_data['val_wer'], label='Valid WER')
        plt.xlabel('Epoch')
        plt.ylabel('WER')
        plt.title('Validation WER')
        plt.legend()
        save_plot('wer', outdir)

    # Learning rate and steps
    fig, ax1 = plt.subplots()
    ax1.plot(epochs, df['lr'], 'g-', label='Learning Rate')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Learning Rate', color='g')

    ax2 = ax1.twinx()
    ax2.plot(epochs, df['steps'], 'b--', label='Steps')
    ax2.set_ylabel('Steps', color='b')
    plt.title('Learning Rate and Steps')
    save_plot('lr_steps', outdir)


def save_plot(basename: str, outdir: str) -> None:
    """Tiny utility: saves current plot as both PNG and PDF, then closes it."""

    png_path = os.path.join(outdir, f'{basename}.png')
    pdf_path = os.path.join(outdir, f'{basename}.pdf')
    plt.savefig(png_path)
    plt.savefig(pdf_path)
    logging.info(f'Saved plot: {png_path}\nSaved plot: {pdf_path}')
    plt.close()


if __name__ == '__main__':
    main()
