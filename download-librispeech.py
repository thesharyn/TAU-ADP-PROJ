import argparse
import logging
import os
import sys
import requests
from speechbrain.utils.data_utils import download_file
from tqdm import tqdm

# URL scheme(s)
URL_PATTERNS = {
    'US': r'https://us.openslr.org/resources/12/',
    'EU': r'https://openslr.elda.org/resources/12/',
    'CN': r'https://openslr.magicdatatech.com/resources/12/'
}

# Default param values
DEFAULT_SAVE_PATH = './datasets/LibriSpeech'
DEF_TRAIN_SPLITS = ["train-clean-100", "train-clean-360", "train-other-500"]
DEF_VAL_SPLITS = ["dev-clean"]
DEF_TEST_SPLITS = ["test-clean", "test-other"]
DEFAULT_SPLITS = DEF_TRAIN_SPLITS + DEF_VAL_SPLITS + DEF_TEST_SPLITS

def retrieve_librispeech(split_names: list[str], outdir: str, overwrite_archives: bool, selected_zone: str, silent: bool) -> None:
    filenames = [split + '.tar.gz' for split in split_names]
    file_urls = [URL_PATTERNS[selected_zone] + file for file in filenames]

    # Preamble: calculate size in advance and see whether to proceed
    total_size = 0
    for url in file_urls:
        resp = requests.head(url)
        file_size = int(resp.headers['Content-Length'])
        total_size += file_size

    print(f'Total size expected to be downloaded ({len(split_names)} archives): {total_size / (1024 ** 3):.3f} GB')
    if overwrite_archives:
        print('Note that I am configured to overwrite the archives if ones with the same names already reside in the folder.')
    else:
        print('In case the archive would be found in the directory now, it will assumed to be correct, hence some bytes won\'t have to be download.')

    if not silent:  # require confirmation to continue
        answer = input('Do you agree to proceed? (Y/N) ').strip().lower()
        while answer not in ('y', 'n', 'yes', 'no'):
            answer = input('Invalid answer. Shall I proceed? (Y/N) ').strip().lower()

        if answer in ('n', 'no'):
            print('Aborting.')
            return


    # Download every archive and extract
    successful = 0
    for file, url in tqdm(zip(filenames, file_urls), len=len(split_names), desc=f'Retrieving each of the {len(split_names)} splits'):
        try:
            archive_path = os.path.join(outdir, file)
            download_file(url, archive_path, replace_existing=overwrite_archives, unpack=True, dest_unpack=outdir)
            successful += 1
        except Exception as e:
            logging.error(f'Error downloading/extracting {file}:' + str(e))
            continue

    if successful == len(split_names):
        print(f'All splits were retrieved - both downloaded and extracted into {outdir}.')
    else:
        print(f'Only some of the splits were successfully retrieved: {successful} / {len(split_names)}.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--splits', nargs='+', default=DEFAULT_SPLITS,
                        help='List of split names to download. E.g., --splits train-clean-100 dev-clean dev-other')
    parser.add_argument('--outdir', type=str, default='./datasets/',
                        help='Directory to stores datasets (or LibriSpeech only) in.')
    parser.add_argument('--overwrite_archives', action='store_true',
                        help='If set, existing archive files will be overwritten instead of being reused. Default is to reuse, but one'
                             'should be sure in that case there is no unrelated/inaccurate file in the folder with the same name.')
    parser.add_argument('--selected_zone', type=str, choices=URL_PATTERNS.keys(), default='EU',
                        help='Which regional mirror server to download from. Options: US, EU, CN. Default is EU (closest to IL).')
    parser.add_argument('--silent', action='store_true',
                        help='If set, the script will not ask for confirmation before proceeding.')

    args = parser.parse_args()
    try:
        os.makedirs(args.outdir, exist_ok=True)
    except (PermissionError, OSError):
        logging.error(f'Directory \'{args.outdir}\' does not exist nor could be created (might be a permissions issue). Aborting.')
        sys.exit(1)
    finally:
        logging.info('Target directory: ' + args.outdir)

    retrieve_librispeech(args.splits, args.outdir)
