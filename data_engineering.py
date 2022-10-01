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
