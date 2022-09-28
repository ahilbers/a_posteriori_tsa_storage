import os
import typing
import logging
import json
import yaml
import shutil
import numpy as np
import pandas as pd
import config
import data_engineering
import models
import aggregation


# Inherited logger, usually created in main.py
logger = logging.getLogger(name='psm')


def get_base_time_series(run_config: dict) -> typing.Tuple[dict, pd.DataFrame]:
    '''Create base time series -- allows benchmarking against multiple 'full' time series'''

    # Load full time series and slice its index and columns
    logger.info('Loading base time series.')
    ts_data_base = models.utils.load_time_series_data(model_name=run_config['model']['model_name'])
    if run_config['ts_base']['column_subset'] is not None:
        ts_data_base = ts_data_base[run_config['ts_base']['column_subset']]
    if run_config['ts_base']['time_slice'] is not None:
        start, end = run_config['ts_base']['time_slice'][0], run_config['ts_base']['time_slice'][1]
        ts_data_base = ts_data_base.loc[start:end]

    # Drop solar columns -- we don't use it in this repo
    ts_data_base = ts_data_base.drop(columns=[i for i in ts_data_base.columns if 'solar' in i])

    # Subsample years randomly to create new base time series (before aggregation)
    if run_config['ts_base']['resample_num_years'] is not None:
        resample_num_years = run_config['ts_base']['resample_num_years']
        logger.debug(f'Creating base time series by resampling {resample_num_years} year(s).')
        run_config, ts_data_base = (
            aggregation.get_years_resample(run_config=run_config, ts_data_full=ts_data_base)
        )

    # 'Roll' DataFrame by putting 'base_ts_roll_days' last days at front -- e.g. start time series
    # in summer -- can be used to minimise impact of empty initial storage level
    if run_config['ts_base']['roll_days'] != 0:
        new_data = np.roll(ts_data_base.to_numpy(), 24 * run_config['ts_base']['roll_days'], axis=0)
        new_index = (
            pd.date_range(start='2021-01-01', periods=ts_data_base.shape[0], freq='h')
            - pd.Timedelta(run_config['ts_base']['roll_days'], unit='d')
        )
        logger.info(f'Rolling time series to new first day. Reset index to start {new_index[0]}.')
        ts_data_base = pd.DataFrame(data=new_data, index=new_index, columns=ts_data_base.columns)

    logger.info(f'Loaded base time series. Length: {round(ts_data_base.shape[0] / 24)} days. ')
    logger.info(f'Start: {ts_data_base.index[0]}. Finish: {ts_data_base.index[-1]}.')

    return run_config, ts_data_base


def update_operate_run_config_with_design_estimate(run_config: dict) -> dict:
    '''Update a run config for a simulation in operate mode: add base time series and fixed
    capacities from a 'get_design_estimate' run.'''

    run_config['model']['run_mode'] = 'operate'

    # Get run config and fixed caps from the design estimate
    sim_id_operate = run_config['simulation']['id']
    sim_id_design = sim_id_operate.replace('get_op', 'get_ds')
    run_config_design = data_engineering.load_run_config(simulation_id=sim_id_design)
    summary_outputs_design = data_engineering.load_summary_outputs(simulation_id=sim_id_design)
    # Float errors cause problems due to a Calliope bug -- hack  # TODO: deal with properly
    summary_outputs_design.loc[summary_outputs_design['output'] < 0.0001] = 0.0001
    cap_regex = '^cap_.*_region\d(_region\d)?$'
    fixed_caps_design = (
        summary_outputs_design
        .filter(regex=cap_regex, axis=0)
        .loc[:, 'output']
        .to_dict()
    )
    if not len(fixed_caps_design) == config.NUMBER_OF_CAPACITIES:
        raise ValueError('Incorrect number of capacities in design vector: check for errors.')

    # Update run config -- use same base time series as design estimate, and update capacities
    run_config['ts_base'] = run_config_design['ts_base']
    run_config['model']['fixed_caps'] = fixed_caps_design  # Use capacities from design estimate
    return run_config


def run_model(run_config: dict, ts_data: pd.DataFrame):
    '''Run energy system optimisation model and save outputs to file.'''

    simulation_id = run_config['simulation']['id']
    summary_outputs_filepath = f'{config.SUMMARY_OUTPUTS_SAVE_DIR}/{simulation_id}.csv'

    if 'cluster' in ts_data.columns:
        ts_aggregation_num_days = run_config['ts_aggregation']['num_days']
        assert ts_aggregation_num_days == ts_data['cluster'].nunique()
        ts_aggregation_num_days = ts_data['cluster'].nunique()
        logger.info(
            f'Running PSM on time series of length {round(ts_data.shape[0] / 24)} days, '
            f'aggregated into {ts_aggregation_num_days} days.'
        )
    else:
        logger.info(f'Running PSM on time series of length {round(ts_data.shape[0] / 24)} days.')
    logger.debug(f'Run config:\n{json.dumps(run_config, indent=4)}.')
    logger.debug(f'Time series data:\n\n{ts_data}\n')

    # Get correct model class
    model_name = run_config['model']['model_name']
    if model_name == '1_region':
        Model = models.models.OneRegionModel
    elif model_name == '6_region':
        Model = models.models.SixRegionModel
    else:
        raise NotImplementedError(f'No model corresponding to name `{model_name}`.')

    # Save run config to file
    if run_config['save']['save_run_config']:
        run_config_filepath = f'{config.RUN_CONFIG_SAVE_DIR}/{simulation_id}.yaml'
        logger.debug(f'Saving run config to {run_config_filepath}.')
        with open(file=run_config_filepath, mode='w') as file:
            yaml.dump(data=run_config, stream=file)

    # Duplicate model directory -- prevents race conditions when runs are done in parallel
    model_name = run_config['model']['model_name']
    model_files_dir = f'{os.path.dirname(__file__)}/model_files'
    orig_model_dir = f'{model_files_dir}/{model_name}'
    new_model_dir = f'{model_files_dir}/{model_name}_{simulation_id}'
    if os.path.exists(new_model_dir):
        logger.debug(f'{new_model_dir} already exists, deleting it and creating a new one.')
        shutil.rmtree(path=new_model_dir)
    shutil.copytree(src=orig_model_dir, dst=new_model_dir)

    logger.info('Creating model.')
    logger.debug(f'Creating model from directory {new_model_dir}.')
    model_config = run_config['model']
    model = Model(
        run_id=simulation_id,
        ts_data=ts_data,
        run_mode=model_config['run_mode'],
        baseload_integer=model_config['baseload_integer'],
        baseload_ramping=model_config['baseload_ramping'],
        allow_unmet=model_config['allow_unmet'],
        fixed_caps=model_config['fixed_caps'] if len(model_config['fixed_caps']) > 0 else None,
        extra_override_name=model_config['extra_override_name']
    )

    # Remove duplicated model directory
    logger.debug(f'Removing new model directory {new_model_dir}.')
    shutil.rmtree(path=new_model_dir)

    logger.info('Running model to determine optimal solution.')
    model.run()
    logger.info('Done running model.')

    # Save outputs to file
    if run_config['save']['save_summary_outputs']:
        logger.debug(f'Saving summary outputs to {summary_outputs_filepath}.')
        model.get_summary_outputs().to_csv(summary_outputs_filepath)
    if run_config['save']['save_ts_outputs']:
        ts_outputs_filepath = f'{config.TS_OUTPUTS_SAVE_DIR}/{simulation_id}.csv'
        logger.debug(f'Saving time series outputs to {ts_outputs_filepath}.')
        model.get_timeseries_outputs().round(3).to_csv(ts_outputs_filepath)

    # Create plots and save to file
    if run_config['save']['save_plot']:
        ts_out = model.get_timeseries_outputs()
        plot_save_filepath = f'{config.PLOT_SAVE_DIR}/{simulation_id}.pdf'

        # Add string to plot with some key summary outputs
        str_columns = [
            'cap_baseload_total',
            'cap_peaking_total',
            'cap_wind_total',
            'cap_storage_energy_total',
            'cap_transmission_total',
            'gen_unmet_total',
            'cost_total',
            'demand_total'
        ]
        summary_outputs = model.get_summary_outputs().loc[str_columns, 'output']
        summary_outputs.index = [i.replace('_total', '') for i in summary_outputs.index]
        summary_outputs_str = yaml.dump(summary_outputs.to_dict())

        # Make an hourly generation level time series to plot
        logger.debug(f'Saving plot to {plot_save_filepath}.')
        ts_to_plot_hourly = ts_out.copy()
        if run_config['save']['plot_ts_slice'] is not None:
            llim, ulim = tuple(run_config['save']['plot_ts_slice'])
            ts_to_plot_hourly = ts_to_plot_hourly[llim:ulim]
        ts_orig_hourly = ts_data.loc[ts_to_plot_hourly.index]

        # Make a daily generation and storage level time series to plot
        ts_to_plot_daily = ts_out.copy().resample('24h').mean()  # Generation levels: mean
        # Storage levels should not be resampled to mean, take first in each day
        st_lvl_cols = ts_out.filter(regex='^level_storage.*$', axis=1).columns
        ts_to_plot_daily[st_lvl_cols] = ts_out.copy()[st_lvl_cols].resample('24h').first()
        ts_to_plot_daily = ts_to_plot_daily.iloc[:365]  # Plot just first year
        ts_orig_daily = ts_data.resample('24h').mean().iloc[:365]

        models.plotting.plot_model_timeseries(
            ts_out_hourly=ts_to_plot_hourly,
            ts_out_daily=ts_to_plot_daily,
            ts_orig_hourly=ts_orig_hourly,
            ts_orig_daily=ts_orig_daily,
            open_plot=False,
            plot_save_filepath=plot_save_filepath,
            annotate_text=summary_outputs_str,
            cap_storage_energy=summary_outputs.loc['cap_storage_energy'],
            ylim_gen=[0, 250]
        )

    # Save all model outputs to file
    if run_config['save']['save_full_outputs']:
        full_outputs_save_dir = f'outputs/full_outputs/{simulation_id}'
        logger.debug(f'Saving full model outputs to {full_outputs_save_dir}.')
        model.to_csv(full_outputs_save_dir)


def get_design_estimate(run_config: dict):
    '''Run a simulation to estimate optimal design, possibly with subsampled data.'''

    # Load base time series, aggregate it, and run planning model
    run_config, ts_data_base = get_base_time_series(run_config=run_config)
    ts_data = aggregation.aggregate_time_series(run_config=run_config, ts_data_full=ts_data_base)
    run_model(run_config=run_config, ts_data=ts_data)


def get_operate_variables(run_config: dict):
    '''Get operational variables by running model with fixed design across full time series.'''

    # Add fixed capacities and base time series from design estimate planning model run
    run_config = update_operate_run_config_with_design_estimate(run_config=run_config)

    # Change some settings for operate mode
    run_config['save']['save_ts_outputs'] = True  # Operate variables in time series -- save them
    run_config['model']['allow_unmet'] = True  # Operate run always allows unmet demand
    run_config['ts_aggregation'] = {
        'method': None, 'column_subset': None, 'num_days': None, 'num_days_extreme': None
    }  # Operate model is run on full time series without any subsampling

    # Get same base time series as design estimate (before sampling) and run operate model
    run_config, ts_data = get_base_time_series(run_config)
    run_model(run_config=run_config, ts_data=ts_data)


def main(run_config: dict):
    '''Conduct design estimation or evaluation run.'''

    simulation_id = run_config['simulation']['id']
    np.random.seed(run_config['simulation']['replication'])

    logger.info(f'Simulation {simulation_id}.')
    simulation_type = run_config['simulation']['type']
    if simulation_type == 'get_design_estimate':
        get_design_estimate(run_config=run_config)
    elif simulation_type == 'get_operate_variables':
        get_operate_variables(run_config=run_config)
    else:
        raise NotImplementedError(f'No action for simulation type `{simulation_type}`.')
    logger.debug('\n\n')  # Whitespace, make debug level log files more readable
