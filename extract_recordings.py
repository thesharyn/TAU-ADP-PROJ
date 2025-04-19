import argparse
import glob
import logging
import os
import random
import sys

import torchaudio

# Drawing instructions:
# samples are taken from directory with the (val) name and are considered of split (key)
SPLITS = {
    'train_clean': 'train-clean-100',
    'train_other': 'train-other-500',
    'val_clean': 'dev-clean',
    'test_clean': 'test-clean',
    'test_other': 'test-other',
}


def extract_samples(librispeech_dir: str, outdir: str, n_samples_per_split: int):
    """
    Extracts audio recordings from the LibriSpeech datasets and saves them as MP3.

    :param librispeech_dir: directory containing the LibriSpeech dataset
    :param outdir: output directory (assumed to exist and be accessible)
    :param n_samples_per_split: how many to draw
    """


    for split_name, subdir in SPLITS.items():
        search_pattern = os.path.join(librispeech_dir, subdir, '**/*.flac')  # looks for .flac files
        flac_files = glob.glob(search_pattern, recursive=True)

        if len(flac_files) < n_samples_per_split:
            # Shouldn't happen (only if a lot of recorded are asked)
            logging.warning(f"[!] Not enough .flac files in {subdir} (found {len(flac_files)}). Skipping.")
            continue

        selected = random.sample(flac_files, n_samples_per_split)

        for i, flac_path in enumerate(selected):
            mp3_name = f"{split_name}_{i}.mp3"
            mp3_path = os.path.join(outdir, mp3_name)

            # Load and convert
            waveform, sr = torchaudio.load(flac_path)
            torchaudio.save(mp3_path, waveform, sr, format='mp3')
            print(f"[✓] Converted {os.path.relpath(flac_path, start=librispeech_path)} → {mp3_name}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extracts sample audio recordings from the LibriSpeech dataset.')
    parser.add_argument('--datasets_dir', type=str, default='./datasets/',
                        help='Directory that contains the LibriSpeech/ dataset folder.')
    parser.add_argument('--outdir', type=str, default='./sample-recordings/',
                        help='Directory to store the example recordings in.')
    parser.add_argument('--num_samples', type=int, default=4,
                        help='Number of recordings to draw (for each splits).')
    args = parser.parse_args()

    librispeech_path = os.path.join(args.datasets_dir, 'LibriSpeech/')
    if not os.path.isdir(librispeech_path):
        logging.error(f'Supposed dataset directory {librispeech_path} does not exist.')
        sys.exit(1)
    elif args.num_samples < 1:
        logging.error(f'Have to draw a positive number of samples; {args.num_samples} is not enough.')
        sys.exit(1)
    elif args.num_samples > 2_000:
        logging.warning(f'Chosen many samples to draw... {args.num_samples} from each split-section.')

    os.makedirs(args.outdir, exist_ok=True)

    extract_samples(librispeech_path, args.outdir, args.num_samples)
    print(f'Done.\nOutput directory: {args.outdir}')
