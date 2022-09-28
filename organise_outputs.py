'''
Organise outputs from model runs. Examples of how to call:

- collate summary output files into single CSV
    `python3 organise_outputs.py collate filename=summary_outputs.csv`
- clean output directory to prepare for new simulations:
    `python3 organise_outputs clean`
'''


import sys
import os
import shutil
import config


def is_valid_summary_output_file(filename: str):
    '''Check whether a filename is a summary output file to be added to master CSV.'''
    if (filename[-4:] != '.csv') or (filename == 'summary_outputs.csv'):
        return False
    return True


def collate_summary_outputs(save_filename: str):
    '''Collate individual summary output files into master and save this to CSV.'''

    import pandas as pd  # Import here -- avoid import errors when using base python

    # Read in each individual file into single master CSV
    summary_output_filenames = sorted([
        i for i in os.listdir(config.SUMMARY_OUTPUTS_SAVE_DIR) if is_valid_summary_output_file(i)
    ])
    collated_results = pd.DataFrame()
    for old_filename in summary_output_filenames:
        filepath = f'{config.SUMMARY_OUTPUTS_SAVE_DIR}/{old_filename}'
        summary_outputs = pd.read_csv(filepath, index_col=0).loc[:, 'output']
        summary_outputs.name = old_filename.split('.')[0]
        collated_results = collated_results.append(summary_outputs)
    collated_results = collated_results.dropna(axis=1)  # Drop columns with NaN, result of HPC bug
    collated_results.to_csv(f'{config.OUTPUTS_DIR}/{save_filename}')


def reverse_collate_summary_outputs(save_filename: str):
    '''Uncollate summary output file into individual iterations and save to CSV.'''

    import pandas as pd  # Import here -- avoid import errors when using base python

    # Read in collated file and save each row as its own CSV
    collated_results_filename = f'{config.OUTPUTS_DIR}/{save_filename}'
    collated_results = pd.read_csv(collated_results_filename, index_col=0)
    for name in collated_results.index:
        if 'benchmark' in name:
            summary_outputs = collated_results.loc[name].to_frame()
            summary_outputs.columns = ['output']
            summary_outputs.to_csv(f'{config.SUMMARY_OUTPUTS_SAVE_DIR}/{name}.csv')


def clean_outputs_dir():
    '''Clear logs subdirectories in outputs directory, ready for new simulations.'''

    # Delete them and recreate empty ones
    dir_to_clear_list = [
        config.LOG_SAVE_DIR,
        config.RUN_CONFIG_SAVE_DIR,
        config.SUMMARY_OUTPUTS_SAVE_DIR,
        config.TS_OUTPUTS_SAVE_DIR,
        config.PLOT_SAVE_DIR
    ]
    for dir_to_clear in dir_to_clear_list:
        if os.path.exists(dir_to_clear):
            shutil.rmtree(dir_to_clear)
        os.mkdir(dir_to_clear)

    # Remove logs
    log_filename_list = [i for i in os.listdir(config.OUTPUTS_DIR) if i[-4:] == '.log']
    for log_filename in log_filename_list:
        os.remove(f'{config.OUTPUTS_DIR}/{log_filename}')


def main():
    '''Organise outputs in desired manner, as specified by command line argument.'''

    # Read in command line argument
    if len(sys.argv) == 1:
        raise ValueError('Must pass second argument to `organise_outputs.py`.')
    command = sys.argv[1]

    # Organise summary outputs into single file
    if command == 'collate':
        if len(sys.argv) == 2:
            raise ValueError('Must pass argument of form `filename=XXX.csv` when collating.')
        save_filename = sys.argv[2].split('=')[1]
        collate_summary_outputs(save_filename=save_filename)

    # Create individual summary output file from master CSV
    if command == 'reverse_collate':
        if len(sys.argv) == 2:
            raise ValueError('Must pass argument of form `filename=XXX.csv` when uncollating.')
        save_filename = sys.argv[2].split('=')[1]
        reverse_collate_summary_outputs(save_filename=save_filename)

    # Clean outputs directory, ready for new simulations
    elif command == 'clean':
        response = input('Clear output directories? Type "y" to confirm: ')
        if response == 'y':
            clean_outputs_dir()
        else:
            print('Cancelled by user -- did nothing.')
            return 0


if __name__ == '__main__':
    main()
