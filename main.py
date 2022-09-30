import os
import copy
import warnings
import argparse
import config
import data_engineering
import models
import simulations


WARNING_MESSAGES_TO_IGNORE = [
    '.*\n.*setting depreciation rate as 1/lifetime.*',  # Warning about no without discount rate
    '.*warmstart.*'  # Warning about warm start for solver in operate mode
]
for warning_message_to_ignore in WARNING_MESSAGES_TO_IGNORE:
    warnings.filterwarnings(action='ignore', message=warning_message_to_ignore)
logger = models.utils.get_logger(name='psm', run_config=config.main_run_config)


def parse_args() -> argparse.Namespace:
    '''Read in model run arguments from bash command. For info, see `config.py`.'''

    # Read in arguments
    parser = argparse.ArgumentParser()
    for argument_name, argument_params in config.COMMAND_LINE_ARGUMENTS.items():
        required = argument_params['required']
        default = argument_params['default']
        parser.add_argument(f'--{argument_name}', required=required, default=default)
    args = parser.parse_args()

    # Convert string arguments to correct data types
    for argument_name, argument_params in config.COMMAND_LINE_ARGUMENTS.items():
        type_to = argument_params['type']
        string_to_convert = getattr(args, argument_name)
        converted_string = data_engineering.convert_string(string_to_convert, type_to=type_to)
        setattr(args, argument_name, converted_string)

    # Check for valid simulation type
    valid_sim_types = ['get_design_estimate', 'get_operate_variables']
    if args.simulation_type not in valid_sim_types:
        raise ValueError(f'Simulation type `{args.simulation_type}` not in {valid_sim_types}.')

    # If extra_config is not specified, it corresponds to 'main' in 'config.py'
    if args.extra_config_name is None:
        logger.debug('Argument `extra_config_name` not specified, mapping to `main`.')
        args.extra_config_name = 'main'

    return args


def create_run_config(main_config: dict) -> dict:
    '''Combine settings into single run config.

    Config settings are updated in following order (last update has highest priority):
    - main_run_config (in `config.py`)
    - command line arguments
    - extra_config (in `config.py`)
    - simulation config (in `config.py`)
    '''

    def update_run_config(run_config: dict, run_config_update_name: str) -> dict:
        '''Update run config via nested dictionary updates.'''
        if not hasattr(config, run_config_update_name):
            raise AttributeError(f'Dict for `{run_config_update_name}` not defined in `config.py`.')
        run_config_update = copy.deepcopy(getattr(config, run_config_update_name))
        for key in run_config.keys():
            if key == 'ts_aggregation':
                # Take special attention with these nested dictionaries
                for deep_key in run_config[key].keys():
                    deep_update = run_config_update.get(key, {}).pop(deep_key, {})
                    if deep_key in ['stratification', 'clustering']:
                        run_config[key][deep_key].update(deep_update)
                    elif deep_update != {}:
                        run_config[key][deep_key] = deep_update
                run_config_update.pop(key, {})
            else:
                run_config[key].update(run_config_update.pop(key, {}))
        assert len(run_config_update) == 0  # Ensure all updates are added
        return run_config

    run_config = copy.deepcopy(main_config)

    # Add command line arguments except for fixed storage cap and subsampling arguments (done below)
    cl_args = {i[0]: i[1] for i in parse_args()._get_kwargs()}  # Read in command line arguments
    run_config['simulation']['name'] = cl_args.pop('simulation_name')
    run_config['simulation']['type'] = cl_args.pop('simulation_type')
    run_config['simulation']['extra_config_name'] = cl_args.pop('extra_config_name')
    run_config['ts_base']['resample_num_years'] = cl_args.pop('ts_base_resample_num_years')
    run_config['simulation']['replication'] = cl_args.pop('replication')

    # Add extra config and simulation config
    run_config = update_run_config(
        run_config=run_config, run_config_update_name=run_config['simulation']['extra_config_name']
    )
    run_config = update_run_config(
        run_config=run_config, run_config_update_name=run_config['simulation']['name']
    )

    # Add time series aggregation settings
    agg_config = run_config['ts_aggregation']
    agg_config['num_days'] = cl_args.pop('ts_reduction_num_days')
    if agg_config['num_days'] is not None:
        if agg_config['stratification']['stratify']:
            column_used = agg_config['stratification']['column_used']
            if column_used == 'max_demand_min_wind':
                agg_config['num_days_extreme'] = 6  # 3 for max demand, 3 for min wind
            else:
                agg_config['num_days_extreme'] = round(
                    agg_config['stratification']['ts_agg_split_extreme'] * agg_config['num_days']
                )
        else:
            agg_config['num_days_extreme'] = 0
    else:
        agg_config['num_days_extreme'] = None

    assert len(cl_args) == 0  # Ensure all command line arguments have been added to `run_config`

    # Add replication and simulation_id
    simulation_id = data_engineering.get_simulation_id(run_config=run_config)
    run_config['simulation']['id'] = simulation_id

    return run_config


def main():
    '''Run the simulations'''

    run_config = create_run_config(main_config=config.main_run_config)
    sim_id, replication = run_config['simulation']['id'], run_config['simulation']['replication']
    logger.info(f'Simulation ID: {sim_id}, replication: {replication}.')
    simulations.main(run_config=run_config)
    logger.info('Done.\n\n')


if __name__ == '__main__':
    logger.debug('New python session starting now.')
    main()
