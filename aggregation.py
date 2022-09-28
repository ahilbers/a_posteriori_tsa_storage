import typing
import logging
import copy
import numpy as np
import pandas as pd
import sklearn.cluster
import data_engineering


# Inherited logger, usually created in main.py
logger = logging.getLogger(name='psm')


def get_daily_vectors(ts_data_hourly: pd.DataFrame) -> pd.DataFrame:
    '''Reshape time series data into one vector per day, with hours as columns.'''

    if not ts_data_hourly.shape[0] % 24 == 0:
        raise IndexError('Time series must have 24 values for each day.')
    num_days = round(ts_data_hourly.shape[0] / 24)

    # Reshape data into daily vectors by pivoting each hour's time steps to columns
    daily_index = ts_data_hourly.resample('D').first().index
    daily_columns = [f'{col}_{hour:02d}' for col in ts_data_hourly.columns for hour in range(24)]
    ts_data_daily = pd.DataFrame(index=daily_index, columns=daily_columns, dtype='float')
    for i, input_column in enumerate(ts_data_hourly.columns):
        column_data = ts_data_hourly.loc[:, input_column].to_numpy()
        ts_data_daily.iloc[:, 24*i:24*(i+1)] = np.reshape(column_data, newshape=(num_days, -1))
    assert (ts_data_daily.index == ts_data_hourly.resample('D').first().index).all()
    assert np.prod(ts_data_daily.shape) == np.prod(ts_data_hourly.shape)
    assert np.allclose(
        ts_data_daily.iloc[0].to_numpy(), ts_data_hourly.iloc[:24].to_numpy().reshape(-1, order='F')
    )  # Check first day has correct values

    return ts_data_daily


def get_hourly_vectors(ts_data_daily: pd.DataFrame) -> pd.DataFrame:
    '''Reshape time series data into one row per hour, inverse of `get_daily_vectors`.'''

    if not ts_data_daily.shape[1] % 24 == 0:
        raise IndexError('Time series must have 24 values for each day.')
    num_days = ts_data_daily.shape[0]

    # Reshape data into daily vectors by pivoting each hour's time steps to columns
    hourly_index = pd.date_range(start=ts_data_daily.index[0], periods=24*num_days, freq='h')
    hourly_columns = [col[:-3] for col in ts_data_daily.columns if col[-3:] == '_00']
    ts_data_hourly = pd.DataFrame(index=hourly_index, columns=hourly_columns, dtype='float')
    for column in ts_data_hourly.columns:
        column_data = ts_data_daily.filter(regex=f'{column}_\d\d').to_numpy()
        ts_data_hourly[column] = np.reshape(column_data, newshape=(-1))
    assert (ts_data_hourly.resample('D').first().index == ts_data_daily.index).all()
    assert np.prod(ts_data_hourly.shape) == np.prod(ts_data_daily.shape)
    assert np.allclose(
        ts_data_hourly.iloc[:24].to_numpy().reshape(-1, order='F'), ts_data_daily.iloc[0].to_numpy()
    )  # Check first day has correct values

    return ts_data_hourly


def get_years_resample(
    ts_data_full: pd.DataFrame, run_config: dict
) -> typing.Tuple[dict, pd.DataFrame]:
    '''Create new sample by resampling years from full time series and reset index.'''

    # Get the sampled years, either from run_config or by random sampling with replacement.
    resample_num_years = run_config['ts_base']['resample_num_years']
    if run_config['ts_base']['resample_years_list'] is not None:
        logger.debug('Using year sample defined in run_config.')
        resample_years_list = np.array(run_config['ts_base']['resample_years_list'])
        if len(resample_years_list) != resample_num_years:
            raise ValueError(
                f'Resample years list length {len(resample_years_list)}, expected {resample_num_years}.'
            )
    else:
        logger.debug('No year sample defined in run_config. Sampling years randomly.')
        years_available = list(ts_data_full.index.year.unique())
        resample_years_list = np.random.choice(years_available, size=resample_num_years, replace=True)
    logger.debug(f'Subsampling years. Years sampled: {resample_years_list}.')

    # Sample the years to create time series and reset index
    ts_out = pd.concat([
        ts_data_full.loc[ts_data_full.index.year == sample_year]
        for sample_year in resample_years_list
    ])
    logger.debug('Resetting index to start 2021-01-01.')
    ts_out.index = pd.date_range(start='2021-01-01', periods=ts_out.shape[0], freq='h')

    # Add sample years into run_config -- use int to make JSON serialisable
    run_config['ts_base']['resample_years_list'] = [int(i) for i in resample_years_list]

    return run_config, ts_out


def normalize_columns(
    ts_data: pd.DataFrame, method: str, columns_to_not_normalize: list = None
) -> pd.DataFrame:
    '''Normalize columns to lie on same scale.'''

    ts_data = ts_data.copy()  # Avoid changes outside function scope

    columns_to_normalize = [i for i in ts_data.columns if i not in columns_to_not_normalize]
    for column in columns_to_normalize:
        # If column has all same values, replace with 0.
        if ts_data[column].std() < 1e-5:
            ts_data[column] = 0.
            continue
        if method == 'z-transform':
            # z-transform: substract mean, divide by standard deviation
            mean, std = ts_data[column].mean(), ts_data[column].std()
            ts_data[column] = (ts_data[column] - mean) / std
            assert np.isclose(ts_data[column].mean(), 0.)
            assert np.isclose(ts_data[column].std(), 1.)
        elif method == 'min-max':
            # min-max rescaling: rescale to lie between 0 and 1
            min, max = ts_data[column].min(), ts_data[column].max()
            ts_data[column] = (ts_data[column] - min) / (max - min)
            assert np.isclose(ts_data[column].min(), 0.)
            assert np.isclose(ts_data[column].max(), 1.)
        elif method is None:
            pass
        else:
            raise NotImplementedError(f'No normalization for method {method}.')

    return ts_data


def add_operate_variables(ts_data: pd.DataFrame, run_config: dict) -> pd.DataFrame:
    '''If operate variables used in stratification or aggregation, add them to time series data.'''

    # Get columns used for either stratification or clustering
    columns_used = run_config['ts_aggregation']['clustering']['columns_used'].copy()
    if run_config['ts_aggregation']['stratification']['stratify']:
        stratify_column_used = run_config['ts_aggregation']['stratification']['column_used']
        if stratify_column_used != 'max_demand_min_wind':
            # Max demand, min wind aggregation is a priori -- no operate variables
            columns_used.append(run_config['ts_aggregation']['stratification']['column_used'])
    columns_missing_from_ts_data = [i for i in columns_used if i not in ts_data.columns]

    # If all columns are already there, return the original data -- we don't need operate variables
    if len(columns_missing_from_ts_data) == 0:
        logger.info('All aggregation columns in time series inputs -- not using operate variables.')
        return ts_data

    logger.info(f'Loading {columns_missing_from_ts_data} from operate variables.')

    # Get run config corresponding to operational model run on a priori scheme
    run_config_operate = copy.deepcopy(run_config)
    mean_or_closest = run_config['ts_aggregation']['representative_day']
    run_config_operate['simulation']['name'] = f'agg_inp_{mean_or_closest}'
    run_config_operate['ts_aggregation']['num_days_extreme'] = 0
    run_config_operate['simulation']['type'] = 'get_operate_variables'
    # TODO: Consider cleaning up all this iteration stuff, and remove 'ground truth' altogether

    # Load operate variables from previous operate model run
    simulation_id_operate = data_engineering.get_simulation_id(run_config=run_config_operate)
    try:
        ts_op_vars = data_engineering.load_time_series_outputs(simulation_id=simulation_id_operate)
    except FileNotFoundError:
        message = f'Failed to locate operate variables for simulation `{simulation_id_operate}`.'
        raise FileNotFoundError(message)

    # If using gen_unmet_total, calculate it by summing regional contributions
    if 'gen_unmet_total' in columns_missing_from_ts_data:
        filter_regex = '^gen_unmet_region\d$'
        ts_op_vars['gen_unmet_total'] = ts_op_vars.filter(regex=filter_regex, axis=1).sum(axis=1)
        print(0)

    # Conduct some checks
    common_columns = [i for i in ts_data.columns if i in ts_op_vars.columns]
    assert len(common_columns) > 0
    assert (ts_op_vars.index == ts_data.index).all()
    for column in common_columns:
        assert np.allclose(ts_data[column], ts_op_vars[column], rtol=0.0, atol=0.1)

    # Slice into columns we use
    columns_missing_from_op_vars = [
        i for i in columns_missing_from_ts_data if i not in ts_op_vars.columns
    ]
    if len(columns_missing_from_op_vars) > 0:
        raise IndexError(f'Columns {columns_missing_from_op_vars} not in operate variables.')
    ts_op_vars_used = ts_op_vars[columns_missing_from_ts_data]

    # Merge 'ts_data' and relevant columns from 'ts_op_vars' into single DataFrame
    logger.debug('Merging time series data and operate variables.')
    logger.debug(f'ts_data:\n{ts_data}:\n\nts_operate_variables:\n{ts_op_vars}\n\n')
    ts_data_merged = (
        pd.merge(left=ts_data, right=ts_op_vars_used, left_index=True, right_index=True)
    )
    assert ts_data_merged.shape[0] == ts_data.shape[0]
    assert ts_data_merged.shape[0] == ts_op_vars.shape[0]
    assert ts_data_merged.columns.is_unique

    return ts_data_merged


def add_is_extreme_day_flag(ts_data_daily: pd.DataFrame, run_config: dict) -> pd.DataFrame:
    '''Add Boolean column, called `is_extreme_day`, to `ts_data_daily`.'''

    ts_data_daily = ts_data_daily.copy()  # Avoid changes outside function scope

    agg_config = run_config['ts_aggregation']
    stratify = agg_config['stratification']['stratify']
    column_stratification = agg_config['stratification']['column_used']
    aggfunc_stratification = agg_config['stratification']['aggfunc']

    if not stratify:
        logger.info('No stratification: no \'extreme\' days, all considered \'regular\'.')
        ts_data_daily['is_extreme_day'] = False  # Every day is 'regular', no 'extreme' days
    elif column_stratification == 'max_demand_min_wind':
        # Days with max demand in each region (3 in total), min wind in each region (3 in total)
        # are considered extreme
        logger.info('Marking days with max regional demand or min regional wind as extreme.')
        columns_max = ['demand_region2', 'demand_region4', 'demand_region5']  # TODO: Don't hardcode
        columns_min = ['wind_region2', 'wind_region5', 'wind_region6']
        extreme_days = []
        for column in columns_max:
            extreme_days.append(
                ts_data_daily.filter(regex=f'^{column}_\d{{2}}', axis=1)
                .apply(aggfunc_stratification, axis=1)
                .idxmax()
            )
        for column in columns_min:
            aggfunc_stratification_used = 'min' if aggfunc_stratification == 'max' else 'sum'
            extreme_days.append(
                ts_data_daily.filter(regex=f'^{column}_\d{{2}}', axis=1)
                .apply(aggfunc_stratification_used, axis=1)
                .idxmin()
            )
        extreme_days = sorted(list(set(extreme_days)))  # Removes duplicates
        ts_data_daily['is_extreme_day'] = ts_data_daily.index.isin(extreme_days)
    elif column_stratification in ['generation_cost', 'gen_unmet_total']:
        logger.info(f'Stratifying based on {aggfunc_stratification} of `{column_stratification}`.')
        if f'{column_stratification}_00' not in ts_data_daily.columns:
            raise IndexError(f'Stratification column `{column_stratification}` not in time series.')
        ts_base_split_extreme = agg_config['stratification']['ts_base_split_extreme']
        ts_base_num_days_extreme = round(ts_base_split_extreme * ts_data_daily.shape[0])
        ranking_variables = (
            ts_data_daily
            .filter(regex=f'^{column_stratification}_\d{{2}}$', axis=1)
            .apply(aggfunc_stratification, axis=1)
        )  # 1-d daily series of stratification column values used to rank days by 'extreme'-ness
        # Rank days by their 'importance'
        ranks = ranking_variables.rank(method='first', ascending=False).astype('int')
        assert ranking_variables.loc[ranks.sort_values().index].is_monotonic_decreasing
        if column_stratification == 'gen_unmet_total':
            # Cap number of 'extreme' days in full time series by number with unmet demand
            has_unmet_demand = (ranking_variables > 0.)
            assert (ranking_variables.loc[has_unmet_demand] > 0.).all()
            num_days_with_unmet_demand = has_unmet_demand.sum()
            # 'Extreme' region of full time series: min of [split, num of days with unmet demand]
            ts_base_num_days_extreme = min(ts_base_num_days_extreme, num_days_with_unmet_demand)
        ts_data_daily['is_extreme_day'] = (ranks <= ts_base_num_days_extreme)
        assert ts_data_daily['is_extreme_day'].sum() == ts_base_num_days_extreme
    else:
        raise NotImplementedError(f'No stratification defined for {column_stratification}')

    # If number of extreme days in aggregate more than in full time series, reduce the first
    ts_base_num_days_extreme = ts_data_daily['is_extreme_day'].sum()
    ts_agg_num_days_extreme = agg_config['num_days_extreme']
    if ts_agg_num_days_extreme > ts_base_num_days_extreme:
        logger.info(
            f'More extreme days in aggregated time series ({ts_agg_num_days_extreme}) than in '
            f'base time series ({ts_base_num_days_extreme}). Decreasing number of extreme days '
            f'in aggregated sample to {ts_base_num_days_extreme}.'
        )
        # Change value in run config, applies outside function scope
        agg_config['num_days_extreme'] = int(ts_base_num_days_extreme)

    return ts_data_daily


def get_vectors_used_to_aggregate(ts_data: pd.DataFrame, run_config: dict) -> pd.DataFrame:
    '''Get vectors used to aggregate: one per day, with operational variables and whether
    each day is 'extreme' or 'regular'.'''

    ts_data = ts_data.copy()  # Avoid changes outside function scope

    column_stratification = run_config['ts_aggregation']['stratification']['column_used']
    columns_clustering = run_config['ts_aggregation']['clustering']['columns_used']
    columns_used = [column_stratification] + columns_clustering
    normalize_method = run_config['ts_aggregation']['clustering']['normalize_method']

    # Slice into time series columns we actually use
    ts_used_hourly = ts_data[[i for i in columns_used if i in ts_data.columns]]

    # Add any columns we use but don't already have from operate variables
    ts_used_hourly = add_operate_variables(ts_data=ts_used_hourly, run_config=run_config)

    # Normalize columns to lie on same scale
    ts_used_hourly = normalize_columns(
        ts_data=ts_used_hourly,
        method=normalize_method,
        columns_to_not_normalize=[column_stratification]
    )

    # Reshape to daily vectors
    ts_used_daily = get_daily_vectors(ts_data_hourly=ts_used_hourly)

    # Add column `is_extreme_day` that indicates whether a day is extreme or not
    ts_used_daily = add_is_extreme_day_flag(ts_data_daily=ts_used_daily, run_config=run_config)

    # If stratification column not used in clustering, remove it
    if column_stratification not in columns_clustering:
        columns_to_remove = [i for i in ts_used_daily.columns if column_stratification in i]
        assert len(columns_to_remove) in [0, 24]
        ts_used_daily = ts_used_daily.drop(columns=columns_to_remove)

    return ts_used_daily


def cluster_stratified(vecs: pd.DataFrame, run_config: dict) -> pd.Series:
    '''Cluster vectors in stratified manner: cluster 'extreme' and 'regular' days separately.

    Parameters:
    -----------
    vecs : vectors to cluster, with columns the components and one extra Boolean column,
        called `is_extreme_day`, which indicates whether that day is 'extreme'
    run_config : run configuration settings

    Returns:
    --------
    clusters : the cluster each day belongs in, as an int, same index as `vecs`
    '''

    num_days = run_config['ts_aggregation']['num_days']
    num_days_extreme = run_config['ts_aggregation']['num_days_extreme']  # 'Extreme' days
    num_days_regular = num_days - num_days_extreme  # 'Regular' days
    assert num_days_regular > 0

    # Stratify into 'extreme' and 'regular' days
    vecs_extreme = vecs.loc[vecs['is_extreme_day']]
    vecs_regular = vecs.loc[~vecs['is_extreme_day']]
    vecs_extreme = vecs_extreme.drop(columns=['is_extreme_day'])
    vecs_regular = vecs_regular.drop(columns=['is_extreme_day'])
    columns_one_per_day = [i.split('_')[-1:] for i in vecs.columns if i.split('_')[-1] == '00']
    logger.debug(f'Clustering based on columns {columns_one_per_day}.')
    logger.debug(f'Daily vectors, extreme:\n{vecs_extreme}\n\n')
    logger.debug(f'Daily vectors, regular:\n{vecs_regular}\n\n')

    # Cluster extreme days
    if num_days_extreme > 0:
        if vecs_extreme.shape[0] < num_days_extreme:
            raise ValueError(
                f'Cannot cluster {vecs_extreme.shape[0]} extreme days to {num_days_extreme}.'
            )
        logger.debug(f'Clustering {vecs_extreme.shape[0]} extreme days into {num_days_extreme}.')
        clusterer_extreme = sklearn.cluster.AgglomerativeClustering(n_clusters=num_days_extreme)
        clusterer_extreme.fit(X=vecs_extreme)
        labels_extreme = clusterer_extreme.labels_
    else:
        labels_extreme = np.array([])
    clusters_extreme = pd.Series(data=labels_extreme, index=vecs_extreme.index)
    assert clusters_extreme.nunique() == num_days_extreme

    # Cluster regular days
    if vecs_regular.shape[0] < num_days_regular:
        raise ValueError(
            f'Cannot cluster {vecs_regular.shape[0]} regular days to {num_days_regular}.'
        )
    logger.debug(f'Clustering {vecs_regular.shape[0]} regular days to {num_days_regular}.')
    clusterer_regular = sklearn.cluster.AgglomerativeClustering(n_clusters=num_days_regular)
    clusterer_regular.fit(X=vecs_regular)
    labels_regular = clusterer_regular.labels_ + num_days_extreme  # Start count at num_days_extreme
    clusters_regular = pd.Series(data=labels_regular, index=vecs_regular.index)
    assert clusters_regular.nunique() == num_days_regular

    # Join extreme and regular days together
    clusters = pd.concat([clusters_extreme, clusters_regular]).sort_index().astype('int')
    clusters = clusters.rename('cluster')  # Give Series a name
    assert (clusters.index == vecs.index).all()
    assert clusters.nunique() == num_days
    logger.debug(f'Mapping from days to clusters:\n\n{clusters}\n')

    return clusters


def get_representative_days(
    ts_data_orig_daily: pd.DataFrame,
    vecs_to_cluster: pd.DataFrame,
    clusters: pd.DataFrame,
    run_config: dict
) -> pd.DataFrame:
    '''Get vectors of representative days for each cluster.

    Parameters:
    -----------
    ts_data_orig_daily : original time series data, reshaped into daily vectors, daily index
    vecs_to_cluster : the vectors used to cluster, usually normalized, daily index
    clusters : mapping from days to int representative day index, daily index
    run_config : run configuration settings

    Returns:
    --------
    representative_days : representative days for each cluster, as daily vectors. index = ints of
        cluster numbers, columns = time series values across for each colum and hour
    '''

    num_days = run_config['ts_aggregation']['num_days']
    representative = run_config['ts_aggregation']['representative_day']  # 'mean' or 'closest'

    # Get representative day for each cluster, same columns as `ts_data_orig_daily`
    representative_day_list = []
    if representative == 'mean':
        # Representative day is cluster mean
        for cluster_num in range(num_days):
            days_in_cluster = ts_data_orig_daily.loc[clusters == cluster_num]
            cluster_mean = days_in_cluster.mean().to_frame(name=cluster_num).T
            representative_day_list.append(cluster_mean)
    elif representative == 'closest':
        # Representative day is real day closest (in normalized space) to cluster mean
        cluster_means_normalized = (
            pd.concat([clusters, vecs_to_cluster], axis=1)
            .drop(columns=['is_extreme_day'])
            .groupby('cluster')
            .mean()
        )  # Cluster means, in normalized space
        for cluster_num in range(num_days):
            cluster_mean = cluster_means_normalized.loc[cluster_num]
            vecs_in_cluster = vecs_to_cluster.loc[clusters == cluster_num]
            assert np.allclose((vecs_in_cluster - cluster_mean).sum(), 0.)
            closest_day_index = (vecs_in_cluster - cluster_mean).pow(2).sum(axis=1).idxmin()
            closest_day = ts_data_orig_daily.loc[closest_day_index].to_frame(name=cluster_num).T
            representative_day_list.append(closest_day)
    else:
        raise NotImplementedError()
    representative_days = pd.concat(representative_day_list)

    return representative_days


def aggregate_into_representative_days(run_config: dict, ts_data: pd.DataFrame) -> pd.DataFrame:
    '''Aggregate time series into smaller number of representative days.

    Parameters:
    -----------
    run_config : run configuration settings
    ts_data : time series data before clustering

    Returns:
    --------
    ts_data_clustered : time series data after clustering. Same time steps as `ts_data`, and all
        columns, along with one new one: `cluster`, an integer index of the cluster. All time
        series values are replaced by their representative values, but order is preserved. Hence,
        all days with same value for `cluster` have same time series values.
    '''

    ts_data_orig = ts_data.copy()
    ts_data_orig_daily = get_daily_vectors(ts_data_hourly=ts_data_orig)

    # Log aggregation details
    ts_base_num_days = ts_data_orig_daily.shape[0]
    ts_agg_num_days = run_config['ts_aggregation']['num_days']  # Number of representative days
    logger.info(f'Aggregating {ts_base_num_days} days to {ts_agg_num_days} representative days.')
    if run_config['ts_aggregation']['stratification']['stratify']:
        stratification_column = run_config['ts_aggregation']['stratification']['column_used']
        logger.info(f'Stratifying time series using column `{stratification_column}`.')
    cluster_columns = run_config['ts_aggregation']['clustering']['columns_used']
    logger.info(f'Clustering time series based on columns {cluster_columns}.')

    # Get vectors used to aggregate: daily, possibly stratified / including operate variables
    vecs_to_cluster = get_vectors_used_to_aggregate(ts_data=ts_data, run_config=run_config)

    # Cluster vectors in stratified manner: get mapping from each day to its cluster
    clusters = cluster_stratified(vecs=vecs_to_cluster, run_config=run_config)
    assert (clusters.index == vecs_to_cluster.index).all()
    assert (clusters.index == ts_data_orig.resample('D').first().index).all()
    assert (clusters.index == ts_data_orig_daily.index).all()
    assert clusters.nunique() == ts_agg_num_days

    # Get representative days for each cluster, one vector per day
    representative_days = get_representative_days(
        ts_data_orig_daily=ts_data_orig_daily,
        vecs_to_cluster=vecs_to_cluster,
        clusters=clusters,
        run_config=run_config
    )
    assert (representative_days.columns == ts_data_orig_daily.columns).all()
    assert (representative_days.index == np.arange(ts_agg_num_days)).all()

    # Replace each day in original time series by cluster representative
    ts_data_aggregated_daily = (
        pd.merge(left=clusters, right=representative_days, left_on='cluster', right_index=True)
    ).sort_index()
    assert (ts_data_aggregated_daily.groupby('cluster').nunique() == 1).all().all()
    assert (ts_data_aggregated_daily.index == ts_data_orig_daily.index).all()
    ts_data_aggregated_daily = ts_data_aggregated_daily.drop(columns=['cluster'])

    # Reshape back to hourly values and add cluster column at hourly resolution
    ts_data_aggregated = get_hourly_vectors(ts_data_daily=ts_data_aggregated_daily)
    ts_data_aggregated['cluster'] = np.repeat(clusters.to_numpy(), 24)

    # Some final checks on aggregated data
    assert (ts_data_aggregated.index == ts_data_orig.index).all()
    assert (ts_data_aggregated.drop(columns=['cluster']).columns == ts_data_orig.columns).all()
    assert ts_data_aggregated.drop_duplicates().shape[0] <= 24 * ts_agg_num_days
    assert (ts_data_aggregated.groupby('cluster').nunique() <= 24).all().all()

    return ts_data_aggregated


def aggregate_time_series(run_config: dict, ts_data_full: pd.DataFrame) -> pd.DataFrame:
    '''Aggregate time series into compressed version with smaller number of representative days.'''

    if run_config['ts_aggregation']['aggregate']:
        if run_config['ts_aggregation']['num_days'] is None:
            raise ValueError('Must specify number of days in aggregated time series.')
        ts_data = aggregate_into_representative_days(run_config=run_config, ts_data=ts_data_full)
    else:
        logger.info('Using base time series without any time series aggregation.')
        ts_data = ts_data_full.copy()

    logger.info('Done creating aggregated time series.')

    return ts_data
