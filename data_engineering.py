'''
Functions for processing data. Can also be called directly:

- collate summary output files into single CSV
    `python3 data_engineering.py collate filename=summary_outputs.csv`
- clean output directory to prepare for new simulations:
    `python3 data_engineering.py clean`
'''


import sys
import os
import shutil
import logging
import yaml
import pandas as pd
import config


# Inherited logger, usually created in main.py
logger = logging.getLogger(name='psm')


def convert_string(string: str, type_to=None):
    '''Map string variable to Python type.'''
    if string in ['FALSE', 'False', 'false']:
        out = False
    elif string in ['TRUE', 'True', 'true']:
        out = True
    elif string in [None, 'NONE', 'None', 'none', 'NULL', 'Null', 'null']:
        out = None
    elif type_to is not None:
        out = type_to(string)
    else:
        out = string
    return out


def load_run_config(simulation_id: str) -> dict:
    filepath = f'{config.RUN_CONFIG_SAVE_DIR}/{simulation_id}.yaml'
    logger.debug(f'Loading run config from {filepath}.')
    with open(file=filepath, mode='r') as file:
        run_config = yaml.safe_load(file)
    return run_config


def load_summary_outputs(simulation_id: str) -> pd.DataFrame:
    filepath = f'{config.SUMMARY_OUTPUTS_SAVE_DIR}/{simulation_id}.csv'
    logger.debug(f'Loading summary outputs from {filepath}.')
    summary_outputs = pd.read_csv(filepath, index_col=0)
    return summary_outputs


def load_time_series_outputs(simulation_id: str) -> pd.DataFrame:
    filepath = f'{config.TS_OUTPUTS_SAVE_DIR}/{simulation_id}.csv'
    logger.debug(f'Loading time series outputs from {filepath}.')
    ts_outputs = pd.read_csv(filepath, index_col=0)
    ts_outputs.index = pd.to_datetime(ts_outputs.index)
    return ts_outputs


def get_simulation_id(run_config: dict) -> str:
    '''Get simulation id for a model run.'''

    simulation_name = run_config['simulation']['name']
    simulation_type = run_config['simulation']['type']
    ts_base_resample_num_years = run_config['ts_base']['resample_num_years']
    ts_agg_num_days = run_config['ts_aggregation']['num_days']
    ts_agg_num_days_extreme = run_config['ts_aggregation']['num_days_extreme']
    replication = run_config['simulation']['replication']

    # Change above variables into strings that become part of simulation id
    simulation_type_dict = {
        'get_design_estimate': 'get_ds',
        'get_operate_variables': 'get_op'
    }
    simulation_name_str = simulation_name
    simulation_type_str = simulation_type_dict[simulation_type]
    resample_num_years_str = (
        f'{ts_base_resample_num_years:02d}y' if ts_base_resample_num_years is not None else 'base'
    )
    replication_str = f'{replication:04d}'

    # Construct simulation id
    base_info = f'{simulation_name_str}--{resample_num_years_str}'
    if simulation_name == 'benchmark':
        simulation_id = f'{base_info}--{replication_str}--{simulation_type_str}'
    elif 'agg' in simulation_name:
        ts_agg_num_days_str = f'{ts_agg_num_days:04d}d'
        ts_agg_num_days_extreme_str = f'{ts_agg_num_days_extreme:03d}dh'
        ts_info = f'{ts_agg_num_days_str}_{ts_agg_num_days_extreme_str}'
        simulation_id = f'{base_info}--{ts_info}--{replication_str}--{simulation_type_str}'
    else:
        raise ValueError(f'Unrecognised simulation name `{simulation_name}`.')

    return simulation_id


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

