'''The test case power system models.'''


import os
import logging
import typing
import json
import numpy as np
import pandas as pd
import calliope
import models.utils


logger = logging.getLogger(name='psm')  # Logger with name 'psm', can be customised elsewhere


class ModelBase(calliope.Model):
    '''Base power system model class.'''

    def __init__(
        self,
        model_name: str,
        ts_data: pd.DataFrame,
        run_mode: str,
        baseload_integer: bool = False,
        baseload_ramping: bool = False,
        allow_unmet: bool = False,
        fixed_caps: dict = None,
        extra_override_name: str = None,
        extra_override_dict: dict = None,
        run_id=0
    ):
        '''
        Create base power system model instance.

        Parameters:
        -----------
        model_name: '1_region' or '6_region'
        ts_data: time series data, can contain custom time step weights
        run_mode: 'plan' or 'operate': optimise capacities or work with prescribed ones
        baseload_integer: enforce discrete baseload capacity constraint (built in discrete units)
        baseload_ramping: enforce baseload ramping constraint
        allow_unmet: allow unmet demand in 'plan' mode (should always be allowed in 'operate' mode)
        fixed_caps: fixed capacities as override
        extra_override: name of additional override, should be defined in  `.../model.yaml`
        extra_override_dict: dictionary with extra override values
        run_id: unique model ID, useful if multiple models are run in parallel
        '''

        if model_name not in ['1_region', '6_region']:
            raise ValueError(f'Invalid model name {model_name} (choose `1_region` or `6_region`).')

        # Some checks when running in operate mode
        if run_mode == 'operate':
            if fixed_caps is None:
                logger.info(
                    f'No fixed capacities passed into model call. '
                    f'Reading fixed capacities from {model_name}/model.yaml'
                )
            if not allow_unmet:
                raise ValueError('Must allow unmet demand when running in operate mode.')

        self.model_name = model_name
        self.run_mode = run_mode
        self.run_id = run_id
        self.base_dir = f'{os.path.dirname(__file__)}/../model_files/{model_name}_{run_id}'
        self.num_timesteps = ts_data.shape[0]

        # Create scenario and override_dict
        scenario = (
            models.utils.get_scenario(run_mode, baseload_integer, baseload_ramping, allow_unmet)
        )
        if extra_override_name is not None:
            scenario = f'{scenario},{extra_override_name}'
        if 'cluster' in ts_data.columns:
            scenario = f'{scenario},clustering'  # set config if cluster specified in time series
        if fixed_caps is not None:
            override_dict = models.utils.get_cap_override_dict(model_name, fixed_caps)
        else:
            override_dict = {}
        if extra_override_dict is not None:
            override_dict.update(extra_override_dict)

        # Get dictionary of time series dataframes and clusters, if applicable
        ts_data, clusters = self._create_init_time_series(ts_data)
        timeseries_dataframes = {'ts_data': ts_data}
        if clusters is not None:
            timeseries_dataframes['clusters'] = clusters  # Not currently used, load from .csv

        # Initialise Calliope model
        super(ModelBase, self).__init__(
            config=os.path.join(self.base_dir, 'model.yaml'),
            timeseries_dataframes=timeseries_dataframes,
            scenario=scenario,
            override_dict=override_dict
        )

        # Adjust time step weights
        if 'weight' in ts_data.columns:
            self.inputs.timestep_weights.values = ts_data.loc[:, 'weight'].values

        logger.debug(f'Model name: {self.model_name}, base directory: {self.base_dir}.')
        logger.debug(f'Override dict:\n{json.dumps(override_dict, indent=4)}')
        logger.debug(f'Time series inputs:\n\n{ts_data}\n')

    def _create_init_time_series(
        self, ts_data: pd.DataFrame
    ) -> typing.Tuple[pd.DataFrame, pd.DataFrame]:
        '''Create time series data for model initialisation.

        Parameters:
        -----------
        ts_data: time series data loaded from CSV

        Returns:
        --------
        ts_data_used: same data, but data engineered for a Calliope model, and with optional column
            'cluster' removed
        clusters: daily clusters, at daily resolution. If no clusters specified, is None.
        '''

        # Avoid changing ts_data outside function
        ts_data_used = ts_data.copy()

        # If cluster specified, split it off into separate DataFrame with daily frequency
        if 'cluster' in ts_data_used.columns:
            clusters_hourly = ts_data_used['cluster']
            if not (clusters_hourly.groupby(clusters_hourly.index.date).nunique() == 1).all():
                raise ValueError('Cluster values should be constant across each day.')
            clusters = clusters_hourly.resample('D').first().to_frame(name='cluster')
            ts_data_used = ts_data_used.drop(columns=['cluster'])
            # For now, save clusters to file and load from there -- implement loading from df later
            clusters.to_csv(f'{self.base_dir}/clusters.csv')
        else:
            clusters = None

        # Detect missing leap days -- reset index if so
        if models.utils.has_missing_leap_days(ts_data_used):  # pragma: no cover
            logger.warning('Missing leap days detected. Time series index reset to start in 2020.')
            ts_data_used.index = pd.date_range(
                start='2020-01-01', periods=self.num_timesteps, freq='h'
            )

        # Demand must be negative for Calliope
        demand_columns = ts_data_used.columns.str.contains('demand')
        ts_data_used.loc[:, demand_columns] = -ts_data_used.loc[:, demand_columns]

        return ts_data_used, clusters

    def run(self):
        '''Run model to determine optimal solution.'''
        logger.debug('Running model to determine optimal solution.')
        super(ModelBase, self).run()
        logger.debug(f'Model summary outputs:\n\n{self.get_summary_outputs()}\n')
        logger.debug(f'Model time series outputs:\n\n{self.get_timeseries_outputs()}\n')
        if not models.utils.has_consistent_outputs(model=self):  # pragma: no cover
            logger.critical('Model has inconsistent outputs. Check log files for details.')
        logger.debug('Done running model.')


class OneRegionModel(ModelBase):
    '''Instance of 1-region power system model.'''

    def __init__(
        self,
        ts_data: pd.DataFrame,
        run_mode: str,
        baseload_integer: bool = False,
        baseload_ramping: bool = False,
        allow_unmet: bool = False,
        fixed_caps: dict = None,
        extra_override_name: str = None,
        extra_override_dict: dict = None,
        run_id=0
    ):
        '''Initialize model from ModelBase parent. See parent class for docstring.'''
        super(OneRegionModel, self).__init__(
            model_name='1_region',
            ts_data=ts_data,
            run_mode=run_mode,
            baseload_integer=baseload_integer,
            baseload_ramping=baseload_ramping,
            allow_unmet=allow_unmet,
            fixed_caps=fixed_caps,
            extra_override_name=extra_override_name,
            extra_override_dict=extra_override_dict,
            run_id=run_id
        )

    def get_summary_outputs(self, as_dict: bool = False) -> typing.Union[pd.DataFrame, dict]:
        '''Return selection of key model outputs: capacities, total generation levels, total demand,
        system cost and carbon emissions.

        Parameters:
        -----------
        as_dict: return dictionary instead of DataFrame
        '''

        if not hasattr(self, 'results'):
            raise AttributeError('Model outputs not yet calculated: call `.run()` first.')

        inp = self.inputs  # Calliope model inputs
        res = self.results  # Calliope model results
        outputs = pd.DataFrame(columns=['output'])  # Output DataFrame to be populated

        # Insert installed capacities
        capacity_info_list = [
            ('cap_baseload_total', 'energy_cap', 'region1::baseload'),
            ('cap_peaking_total', 'energy_cap', 'region1::peaking'),
            ('cap_wind_total', 'resource_area', 'region1::wind'),
            ('cap_solar_total', 'resource_area', 'region1::solar'),
            ('cap_storage_energy_total', 'storage_cap', 'region1::storage_'),
            ('cap_storage_power_total', 'energy_cap', 'region1::storage_')
        ]
        # If this technology is part of the model, add it to the outputs, else set value 0
        for outputs_idx, results_attr_name, results_key in capacity_info_list:
            try:
                results_attr = getattr(res, results_attr_name)
                outputs.loc[outputs_idx] = float(results_attr.loc[results_key])
            except (KeyError, AttributeError):
                outputs.loc[outputs_idx] = 0.  # If this tech isn't in model, set value 0

        # Insert peak unmet demand
        outputs.loc['peak_unmet_total'] = float(res.carrier_prod.loc['region1::unmet::power'].max())

        # Insert generation levels using same logic as above
        for tech in ['baseload', 'peaking', 'wind', 'solar', 'unmet']:
            try:
                outputs.loc[f'gen_{tech}_total'] = float(
                    (res.carrier_prod.loc[f'region1::{tech}::power'] * inp.timestep_weights).sum()
                )
            except KeyError:
                outputs.loc[f'gen_{tech}_total'] = 0.  # If this tech isn't in model, set value 0

        # Insert demand levels
        outputs.loc['demand_total'] = -float(
            (res.carrier_con.loc['region1::demand_power::power'] * inp.timestep_weights).sum()
        )

        # Insert total system cost and carbon emissions
        outputs.loc['cost_total'] = float(self.results.cost.loc[{'costs': 'monetary'}].sum())
        outputs.loc['emissions_total'] = float(self.results.cost.loc[{'costs': 'emissions'}].sum())

        # Insert metadata about optimisation problem
        outputs.loc['num_ts'] = len(self.inputs.timesteps)
        outputs.loc['sum_ts_weights'] = float(sum(self.inputs.timestep_weights))
        outputs.loc['solution_time'] = float(self.results.solution_time)

        if as_dict:
            outputs = outputs['output'].to_dict()

        return outputs

    def get_timeseries_outputs(self, include_final_storage_level: bool = False) -> pd.DataFrame:
        '''Get generation and storage levels for each time step.

        Parameters:
        -----------
        include_final_storage_level: add extra row with the storage levels at the end of the time
            series. Leave all generation columns (with units like MW, not MWh) blank -- fill NaNs
        '''

        if not hasattr(self, 'results'):
            raise AttributeError('Model outputs not yet calculated: call `.run()` first.')

        inp = self.inputs  # Calliope model inputs
        res = self.results  # Calliope model results
        # ts_outputs: output DataFrame to be populated
        ts_outputs = pd.DataFrame(index=pd.to_datetime(inp.timesteps.values))

        # Add time step weight and demand levels
        ts_outputs['ts_weight'] = inp.timestep_weights.values
        ts_outputs['demand'] = - res.carrier_con.loc['region1::demand_power::power'].values

        # Add each generation technology to outputs if part of model, else set value 0
        for tech in ['baseload', 'peaking', 'wind', 'solar', 'unmet']:
            if f'region1::{tech}::power' in res.carrier_prod.loc_tech_carriers_prod:
                ts_outputs[f'gen_{tech}'] = res.carrier_prod.loc[f'region1::{tech}::power'].values
            else:
                ts_outputs[f'gen_{tech}'] = 0.

        # Do same for storage technologies. Without clustering, storage levels are 'intraday'
        # values. With clustering, they are sum of 'intraday' and 'interday', which is added below
        if 'region1::storage_::power' in res.carrier_prod.loc_tech_carriers_prod:
            ts_outputs['gen_storage'] = (
                res.carrier_prod.loc['region1::storage_::power'].values
                + res.carrier_con.loc['region1::storage_::power'].values
            )
            # Storage levels should reflect those at beginning of time step -- so offset by 1
            ts_outputs['level_storage_intraday'] = np.concatenate((
                np.array([
                    inp.storage_initial.loc['region1::storage_']
                    * res.storage_cap.loc['region1::storage_']
                ]),
                res.storage.loc['region1::storage_'].values[:-1]
            ))  # Last storage level is not included in ts_outputs
        else:
            # If no storage technologies, set values to 0
            ts_outputs['gen_storage'] = 0.
            ts_outputs['level_storage_intraday'] = 0.

        # If storage splits into intraday and interday levels, deal with interday here
        # If clustered, expand shortened time series (with each representative day occuring once)
        # to repeating the representative days in the right place
        if hasattr(inp, 'lookup_datestep_cluster'):
            cluster_sequence = inp.lookup_datestep_cluster.values
            num_days = cluster_sequence.shape[0]
            ts_days = []  # Populate with full time series by repeating clusters
            for cluster_num in cluster_sequence:
                cluster_day = ts_outputs.iloc[inp.timestep_cluster.values == cluster_num].copy()
                cluster_day['cluster'] = cluster_num
                cluster_day['ts_weight'] = 1.  # Reset time step weight since we repeat days
                ts_days.append(cluster_day)
            ts_outputs = pd.concat(ts_days)
            # Move 'cluster' column to front of DataFrame, and reset index
            ts_outputs = ts_outputs[['cluster', *ts_outputs.drop(columns='cluster').columns]]
            ts_outputs.index = (
                np.repeat(inp.lookup_datestep_cluster.datesteps.values, 24)
                + np.tile(pd.timedelta_range(start=0, periods=24, freq='h'), num_days)
            )

        # If clustering, deal with the inter-day storage levels here -- they are not repeated
        # across clusters so have to come after the repeating done above
        try:
            storage_inter_cluster_daily = res.storage_inter_cluster.loc['region1::storage_'].values
            ts_outputs['level_storage_interday'] = (
                np.repeat(storage_inter_cluster_daily, 24)  # Upsampled to hourly
                * np.tile(
                    np.power(1 - float(inp.storage_loss), np.arange(24)),
                    storage_inter_cluster_daily.shape[0]
                )  # Hourly storage remaining, from self loss
            )
            # Reset intraday storage level at beginning of day to 0 -- covered by interday level
            ts_outputs.loc[ts_outputs.index.hour == 0, 'level_storage_intraday'] = 0.
        except AttributeError:
            # If no inter-cluster storage (i.e. no decomposition into days), set values to 0
            ts_outputs['level_storage_interday'] = 0.

        # Add storage levels at end of final time step
        if include_final_storage_level:
            t = ts_outputs.index[-1] + pd.Timedelta(1, unit='h')  # Time step at end of time series
            final_storage_interday = (
                (1 - float(inp.storage_loss)) * ts_outputs.iloc[-1]['level_storage_interday']
            )
            cluster_num_last_timestep = inp.lookup_datestep_cluster.values[-1]
            idx_last_timestep = 24 * (cluster_num_last_timestep + 1) - 1
            final_storage_intraday = res.storage.loc['region1::storage_'].values[idx_last_timestep]
            ts_outputs.loc[t, 'level_storage_interday'] = final_storage_interday
            ts_outputs.loc[t, 'level_storage_intraday'] = final_storage_intraday

        ts_outputs['level_storage'] = (
            ts_outputs['level_storage_interday'] + ts_outputs['level_storage_intraday']
        )

        # Add generation costs
        gen_costs_per_unit = inp.cost_om_prod.loc['monetary']
        ts_outputs['generation_cost'] = 0.
        for tech in ['baseload', 'peaking', 'wind', 'solar', 'unmet']:
            if f'region1::{tech}' in gen_costs_per_unit.loc_techs_om_cost:
                ts_outputs['generation_cost'] += (
                    float(gen_costs_per_unit.loc[f'region1::{tech}']) * ts_outputs[f'gen_{tech}']
                )

        # Add carbon emissions
        emissions_per_unit = inp.cost_om_prod.loc['emissions']
        ts_outputs['emissions'] = 0.
        for tech in ['baseload', 'peaking', 'wind', 'solar', 'unmet']:
            if f'region1::{tech}' in emissions_per_unit.loc_techs_om_cost:
                ts_outputs['emissions'] += (
                    float(emissions_per_unit.loc[f'region1::{tech}']) * ts_outputs[f'gen_{tech}']
                )

        return ts_outputs


class SixRegionModel(ModelBase):
    '''Instance of 6-region power system model.'''

    def __init__(
        self,
        ts_data: pd.DataFrame,
        run_mode: str,
        baseload_integer: bool = False,
        baseload_ramping: bool = False,
        allow_unmet: bool = False,
        fixed_caps: dict = None,
        extra_override_name: str = None,
        extra_override_dict: dict = None,
        run_id=0
    ):
        '''Initialize model from ModelBase parent. See parent class for docstring.'''
        super(SixRegionModel, self).__init__(
            model_name='6_region',
            ts_data=ts_data,
            run_mode=run_mode,
            baseload_integer=baseload_integer,
            baseload_ramping=baseload_ramping,
            allow_unmet=allow_unmet,
            fixed_caps=fixed_caps,
            extra_override_name=extra_override_name,
            extra_override_dict=extra_override_dict,
            run_id=run_id
        )

    def get_summary_outputs(self, as_dict: bool = False) -> typing.Union[pd.DataFrame, dict]:
        '''Return selection of key model outputs: capacities, total generation levels, total demand,
        system cost and carbon emissions.

        Parameters:
        -----------
        as_dict: return dictionary instead of DataFrame
        '''

        if not hasattr(self, 'results'):
            raise AttributeError('Model outputs not yet calculated: call `.run()` first.')

        inp = self.inputs  # Calliope model inputs
        res = self.results  # Calliope model results
        tech_regions = models.utils.get_tech_regions(model=self)  # List of tech-region pairs

        outputs = pd.DataFrame(columns=['output'])  # Output DataFrame to be populated

        # Add regional summary outputs to outputs DataFrame
        for tech_region in tech_regions:
            tech, region = tech_region[:2]

            # Add capacities
            if tech in ['baseload', 'peaking']:
                key = f'{region}::{tech}_{region}'
                outputs.loc[f'cap_{tech}_{region}'] = float(res.energy_cap.loc[key])
                outputs.loc[f'gen_{tech}_{region}'] = float()
            elif tech in ['wind']:
                key = f'{region}::{tech}_{region}'
                outputs.loc[f'cap_{tech}_{region}'] = float(res.resource_area.loc[key])
            elif tech in ['transmission']:
                region_to = tech_region[2]
                key = f'{region}::{tech}_{region}_{region_to}:{region_to}'
                assert int(region_to[-1]) > int(region[-1])  # One direction, no double counting
                outputs.loc[f'cap_{tech}_{region}_{region_to}'] = float(res.energy_cap.loc[key])
            elif tech in ['storage']:
                key = f'{region}::{tech}_{region}'
                outputs.loc[f'cap_{tech}_energy_{region}'] = float(res.storage_cap.loc[key])
                outputs.loc[f'cap_{tech}_power_{region}'] = float(res.energy_cap.loc[key])
            elif tech in ['unmet']:
                key = f'{region}::{tech}_{region}::power'
                outputs.loc[f'peak_{tech}_{region}'] = float(res.carrier_prod.loc[key].max())
            else:
                raise ValueError(f'Cannot add capacity for {tech}.')

            # Add generation levels
            if tech in ['baseload', 'peaking', 'wind', 'unmet']:
                key = f'{region}::{tech}_{region}::power'
                outputs.loc[f'gen_{tech}_{region}'] = (
                    float((res.carrier_prod.loc[key] * inp.timestep_weights).sum())
                )

            # Add demand, which is in same regions as 'unmet' tech
            if tech in ['unmet']:
                key = f'{region}::demand_power::power'
                outputs.loc['demand_{}'.format(region)] = (
                    -float((res.carrier_con.loc[key] * inp.timestep_weights).sum())
                )

        # Add total capacities and peak unmet demand
        for tech in [
            'baseload', 'peaking', 'wind', 'storage_energy', 'storage_power', 'transmission'
        ]:
            filter_regex = f'^cap_{tech}_.*$'
            outputs.loc[f'cap_{tech}_total'] = outputs.filter(regex=filter_regex, axis=0).sum()
        outputs.loc['peak_unmet_total'] = outputs.filter(regex='^peak_unmet_.*$', axis=0).sum()

        # Add systemwide peak unmet demand -- not necessarily equal to peak_unmet_total. Total unmet
        # capacity sums peak unmet demand across regions, this is peak at same time
        outputs.loc['peak_unmet_systemwide'] = float(
            res.carrier_prod
            .loc[self.results.carrier_prod.loc_tech_carriers_prod.str.contains('unmet')]
            .sum(axis=0)
            .max()
        )

        # Insert total generation and unmet demand levels
        for tech in ['baseload', 'peaking', 'wind', 'solar', 'unmet']:
            filter_regex = f'^gen_{tech}_.*$'
            outputs.loc[f'gen_{tech}_total'] = outputs.filter(regex=filter_regex, axis=0).sum()

        # Insert total demand levels, emissions, system cost
        num_ts = len(self.inputs.timesteps)
        sum_ts_weights = float(sum(self.inputs.timestep_weights))
        outputs.loc['demand_total'] = outputs.filter(regex='^demand.*', axis=0).sum()
        if self.run_mode == 'plan':
            outputs.loc['emissions_total'] = float(self.results.cost.loc[{'costs': 'emissions'}].sum())
            outputs.loc['cost_install'] = float(res.cost_investment.loc[{'costs': 'monetary'}].sum())
            outputs.loc['cost_operate'] = float(res.cost_var.loc[{'costs': 'monetary'}].sum())
        elif self.run_mode == 'operate':
            # outputs.loc['emissions_total'] = (
            #     float(self.results.cost.loc[{'costs': 'emissions'}].sum()) / sum_ts_weights
            # )  # TODO: Sort out what's happening here
            outputs.loc['cost_install'] = 0.
            outputs.loc['cost_operate'] = float(res.cost_var.loc[{'costs': 'monetary'}].sum())
        outputs.loc['cost_total'] = outputs.loc['cost_install'] + outputs.loc['cost_operate']

        # Insert metadata about optimisation problem
        outputs.loc['num_ts'] = num_ts
        outputs.loc['sum_ts_weights'] = sum_ts_weights
        outputs.loc['solution_time'] = float(self.results.solution_time)

        # Deal with floating point errors around zero
        cap_threshold, gen_threshold, emit_threshold = 1e-3, 1e1, 1e3
        # Check for negative values with magnitude over the threshold
        has_negative_caps = not (outputs['output'].filter(regex='^cap_.*$') > -cap_threshold).all()
        has_negative_gens = not (outputs['output'].filter(regex='^gen_.*$') > -gen_threshold).all()
        if has_negative_caps or has_negative_gens:
            # In this case, return original with negative values, for diagnosis
            logger.error('Some negative capacities or generation levels. Check output file.')
        else:
            # Map values with absolute value less than threshold to 0
            outputs.loc[
                (outputs.index.str.contains('cap_')) & (outputs['output'] < cap_threshold)
            ] = 0.
            outputs.loc[
                (outputs.index.str.contains('gen_')) & (outputs['output'] < gen_threshold)
            ] = 0.
            outputs.loc[
                (outputs.index.str.contains('emissions_')) & (outputs['output'] < emit_threshold)
            ] = 0.
            outputs = outputs.clip(lower=0.)  # Clip small negative values

        if as_dict:
            outputs = outputs['output'].to_dict()

        return outputs

    def get_timeseries_outputs(self, include_final_storage_level: bool = False) -> pd.DataFrame:
        '''Get generation, transmission and storage levels for each time step.

        Parameters:
        -----------
        include_final_storage_level: add extra row with the storage levels at the end of the time
            series. Leave all generation columns (with units like MW, not MWh) blank -- fill NaNs
        '''

        if not hasattr(self, 'results'):
            raise AttributeError('Model outputs not yet calculated: call `.run()` first.')

        inp = self.inputs  # Calliope model inputs
        res = self.results  # Calliope model results
        gen_costs_per_unit = inp.cost_om_prod.loc['monetary']
        emissions_per_unit = inp.cost_om_prod.loc['emissions']
        tech_regions = models.utils.get_tech_regions(model=self)  # List of tech-region pairs
        is_aggregated = hasattr(inp, 'lookup_datestep_cluster')  # Whether time series aggregated

        ts_outputs = pd.DataFrame(index=pd.to_datetime(inp.timesteps.values))  # To be populated
        # Add time step weight and initialise generation cost and emissions, which we build up
        ts_outputs['ts_weight'] = inp.timestep_weights.values
        ts_outputs['generation_cost'] = 0.
        ts_outputs['emissions'] = 0.

        # Add regional summary outputs to outputs DataFrame
        for tech_region in tech_regions:
            tech, region = tech_region[:2]

            # Add demand, which is in same regions as 'unmet' tech
            if tech in ['unmet']:
                key = f'{region}::demand_power'
                ts_outputs[f'demand_{region}'] = - res.carrier_con.loc[f'{key}::power'].values

            # Generation, transmission and storage levels
            if tech in ['baseload', 'peaking', 'wind', 'unmet']:
                key = f'{region}::{tech}_{region}'
                ts_outputs[f'gen_{tech}_{region}'] = res.carrier_prod.loc[f'{key}::power'].values
                ts_outputs['generation_cost'] += (
                    float(gen_costs_per_unit.loc[key]) * ts_outputs[f'gen_{tech}_{region}']
                )
                ts_outputs['emissions'] += (
                    float(emissions_per_unit.loc[key]) * ts_outputs[f'gen_{tech}_{region}']
                )
            elif tech in ['transmission']:
                region_to = tech_region[2]
                key = f'{region}::{tech}_{region}_{region_to}:{region_to}'
                assert int(region_to[-1]) > int(region[-1])  # One direction, no double counting
                key_forward = f'{region}::transmission_{region}_{region_to}:{region_to}::power'
                key_reverse = f'{region_to}::transmission_{region}_{region_to}:{region}::power'
                net_transmission = (
                    - res.carrier_prod.loc[key_forward].values
                    + res.carrier_prod.loc[key_reverse].values
                )  # Net transmission levels into this node
                ts_outputs[f'transmission_{region}_{region_to}'] = net_transmission
            elif tech in ['storage']:
                # Without clustering, storage levels are 'intraday' values. With clustering,
                # they are sum of 'intraday' and 'interday', which is added below
                key = f'{region}::storage_{region}'
                key_power = f'{key}::power'
                ts_outputs[f'gen_storage_{region}'] = (
                    res.carrier_prod.loc[key_power].values + res.carrier_con.loc[key_power].values
                )
                ts_outputs[f'level_storage_interday_{region}'] = np.nan  # Populate later
                # Storage levels should reflect those at beginning of time step -- so offset by 1
                ts_outputs[f'level_storage_intraday_{region}'] = np.concatenate((
                    np.array([inp.storage_initial.loc[key] * res.storage_cap.loc[key]]),
                    res.storage.loc[key].values[:-1]
                ))  # Last storage level is not included in ts_outputs
                ts_outputs[f'level_storage_{region}'] = np.nan  # Populate later

            else:
                raise ValueError(f'Cannot add time series values for {tech}.')

        # If aggregated (clustered), expand aggregated time series (in which each representative
        # day occurs once) to full time series, with representative days repeated in order
        if is_aggregated:
            cluster_sequence = inp.lookup_datestep_cluster.values
            num_days = cluster_sequence.shape[0]
            ts_days = []  # Populate with full time series by repeating clusters
            for cluster_num in cluster_sequence:
                cluster_day = ts_outputs.iloc[inp.timestep_cluster.values == cluster_num].copy()
                cluster_day['cluster'] = cluster_num
                cluster_day['ts_weight'] = 1.  # Representative days in order -- so all weight 1
                ts_days.append(cluster_day)
            ts_outputs = pd.concat(ts_days)
            # Move 'cluster' column to front of DataFrame, and reset index
            ts_outputs = ts_outputs[['cluster', *ts_outputs.drop(columns='cluster').columns]]
            ts_outputs.index = (
                np.repeat(inp.lookup_datestep_cluster.datesteps.values, 24)
                + np.tile(pd.timedelta_range(start=0, periods=24, freq='h'), num_days)
            )

        # If clustering, deal with the inter-day storage levels here -- they are not repeated
        # across clusters so have to come after the repeating done above
        storage_regions = [i[1] for i in tech_regions if i[0] == 'storage']
        for region in storage_regions:
            self_loss = float(inp.storage_loss.loc[key])  # Proportion of charge lost per hour
            key = f'{region}::storage_{region}'
            if is_aggregated:
                storage_inter_cluster_daily = res.storage_inter_cluster.loc[key]
                num_days = storage_inter_cluster_daily.shape[0]
                ts_outputs[f'level_storage_interday_{region}'] = (
                    np.repeat(storage_inter_cluster_daily, 24)  # Upsampled to hourly
                    * np.tile(np.power(1 - self_loss, np.arange(24)), num_days)  # From self loss
                )
                # Reset intraday storage level at beginning of day to 0 -- covered by interday level
                ts_outputs.loc[ts_outputs.index.hour == 0, f'level_storage_intraday_{region}'] = 0.
            else:
                ts_outputs[f'level_storage_interday_{region}'] = 0.

            ts_outputs[f'level_storage_{region}'] = (
                ts_outputs[f'level_storage_interday_{region}']
                + ts_outputs[f'level_storage_intraday_{region}']
            )

        # Add storage levels at end of final time step
        if include_final_storage_level:
            t = ts_outputs.index[-1] + pd.Timedelta(1, unit='h')  # Time step at end of time series
            num_ts = ts_outputs.shape[0]
            if is_aggregated:
                cluster_num_last_timestep = inp.lookup_datestep_cluster.values[-1]
                idx_last_timestep = 24 * (cluster_num_last_timestep + 1) - 1
            for region in storage_regions:
                final_storage_interday = (
                    (1 - self_loss) * ts_outputs.iloc[num_ts-1][f'level_storage_interday_{region}']
                )
                final_storage_intraday = (
                    res.storage.loc[key].values[idx_last_timestep] if is_aggregated else 0.
                )
                final_storage = final_storage_interday + final_storage_intraday
                ts_outputs.loc[t, f'level_storage_interday_{region}'] = final_storage_interday
                ts_outputs.loc[t, f'level_storage_intraday_{region}'] = final_storage_intraday
                ts_outputs.loc[t, f'level_storage_{region}'] = final_storage

        return ts_outputs
