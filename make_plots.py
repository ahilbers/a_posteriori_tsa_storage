import typing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors
import config


# Plot settings
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams["mathtext.rm"] = "serif"
plt.rcParams["mathtext.bf"] = "serif:bold"
plt.rcParams["mathtext.default"] = "regular"
COLOR_GREY = matplotlib.colors.ColorConverter.to_rgba('grey', alpha=0.5)
COLOR_BLUE = matplotlib.colors.ColorConverter.to_rgba('cornflowerblue', alpha=0.6)
EDGE_GREY = matplotlib.colors.ColorConverter.to_rgba('grey', alpha=1)
EDGE_BLUE = matplotlib.colors.ColorConverter.to_rgba('royalblue', alpha=1)
EDGE_METHOD = ['#0A029B', '#0434B6', '#2184EE', '#4EA2C7', '#57B7B6', '#439044']
COLORS_METHOD = [matplotlib.colors.ColorConverter.to_rgba(i, alpha=0.6) for i in EDGE_METHOD]

# Data engineering settings
DEVIATION_COLUMNS = [
    'cap_baseload_total',
    'cap_peaking_total',
    'cap_wind_total',
    'cap_transmission_total',
    'cap_storage_energy_total',
    'cost_total'
]
ZERO_CLIP = 1e-3  # Clip any values less than this when dividing -- avoid division by zero


def _isclose(
    x: pd.DataFrame, y: pd.DataFrame, atol: float = 1e-2, rtol: float = 1e-5
) -> pd.DataFrame:
    '''Check if output DataFrames are close componentwise'''
    isclose_np = np.isclose(x, y, atol=atol, rtol=rtol)
    isclose_pd = pd.DataFrame(data=isclose_np, index=x.index, columns=x.columns)
    return isclose_pd


def _sim_id_to_num_days(sim_id):
    '''Get sample length (in days) from simulation id string.'''
    sim_id_list = sim_id.split('--')
    if sim_id_list[0] == 'benchmark':
        num_days, num_days_extreme = 0, 0
    else:
        num_days_str_combined = sim_id_list[2]
        num_days_str, num_days_extreme_str = num_days_str_combined.split('_')
        num_days = int(num_days_str[:-1])
        num_days_extreme = int(num_days_extreme_str[:-2])

    return num_days, num_days_extreme


def _beautify_boxes(box, colors='blue'):
    '''Color and stylise boxes for box and whiskers plot. Note: modifies boxes in-place.'''
    num_boxes = len(box['boxes'])
    for i in range(num_boxes):
        if colors == 'blue':
            color_face = COLOR_BLUE
            color_edge = EDGE_BLUE
        elif colors == 'gray_and_blue':
            color_face = COLOR_GREY if i % 2 == 0 else COLOR_BLUE
            color_edge = EDGE_GREY if i % 2 == 0 else EDGE_BLUE
        elif colors == 'methods':
            color_face = COLORS_METHOD[-1] if i == num_boxes - 1 else COLORS_METHOD[i]
            color_edge = EDGE_METHOD[-1] if i == num_boxes - 1 else EDGE_METHOD[i]
        else:
            raise NotImplementedError()
        box['boxes'][i].set_color(color_face)
        box['boxes'][i].set_edgecolor(color_edge)
        box['boxes'][i].set_linewidth(2)
        box['whiskers'][2*i].set_color(color_edge)
        box['whiskers'][2*i+1].set_color(color_edge)
        box['whiskers'][2*i].set_linewidth(2)
        box['whiskers'][2*i+1].set_linewidth(2)
        box['caps'][2*i].set_color(color_edge)
        box['caps'][2*i+1].set_color(color_edge)
        box['caps'][2*i].set_linewidth(2)
        box['caps'][2*i+1].set_linewidth(2)
        box['medians'][i].set_color(color_edge)
        box['medians'][i].set_linewidth(2)


def _get_legend_handles(ax, labels, break_long_lines=False) -> list:
    '''Get handles for a legend that labels aggregation methods.'''

    x = np.array([1, 2])
    y = np.array([-200, -200])
    old_ylim = ax.get_ylim()  # Get ylim before adding fake data used for legend
    handles = []
    num_labels = len(labels)
    for i in range(num_labels):
        color = COLORS_METHOD[-1] if i == num_labels - 1 else COLORS_METHOD[i]
        edgecolor = EDGE_METHOD[-1] if i == num_labels - 1 else EDGE_METHOD[i]
        label = labels[i]
        # Plots do not actually appear -- off the plot, just to get correct legend labels
        pl = ax.fill_between(x, 2*y, y, color=color, edgecolor=edgecolor, lw=2, label=label)
        handles.append(pl)
    ax.set_ylim(old_ylim)

    return handles


def _plot_metrics_fixed_subsample_size_across_methods(
    ax, quantiles, col_name, reduction_method_list, ylabel=None, ylim=None, title=None
):

    # Put quantiles on index and method on columns, reordered to match reduction_method_list
    quantiles_plot = quantiles.loc[:, col_name].unstack().T.loc[:, reduction_method_list]

    if 'deviation' in col_name or 'proportion' in col_name:
        quantiles_plot *= 100.  # Express as percentage

    assert (quantiles_plot.columns == reduction_method_list).all()
    box = ax.boxplot(quantiles_plot, whis=(0, 100), widths=0.5, patch_artist=True)
    ax.axhline(y=0, color='black', linestyle='--')
    _beautify_boxes(box=box, colors='methods')
    if ylim is not None:
        ax.set_ylim(ylim)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=9)
    if title is not None:
        ax.set_title(title, fontsize=9)


def _read_outputs(
    num_years_base: int, drop_regional: bool = True
) -> typing.Tuple[pd.DataFrame, pd.DataFrame]:
    '''Read collated summary outputs across replications.'''

    # Read in summary outputs and slice into correct base time series
    out_full = pd.read_csv(f'{config.OUTPUTS_DIR}/summary_outputs.csv', index_col=0).sort_index()
    out_full = out_full.filter(regex=f'^.*--{num_years_base:02d}y--.*$', axis=0)

    # Clean up columns -- use only systemwide totals if `drop_regional`
    columns_to_drop = ['peak_unmet_total', *[i for i in out_full.columns if 'storage_power' in i]]
    out_full = out_full.drop(columns=columns_to_drop)
    if drop_regional:
        columns_to_drop = [i for i in out_full.columns if 'region' in i]
        out_full = out_full.drop(columns=columns_to_drop)

    # Remove replications for which not all simulations are done, in case some HPC runs didn't finish
    out_full['replication'] = [int(i[-2]) for i in out_full.index.str.split('--')]
    sims_per_iter = out_full['replication'].value_counts()
    sims_ignored = sorted(list(sims_per_iter[sims_per_iter < sims_per_iter.max()].index))
    out_full = out_full.loc[~out_full['replication'].isin(sims_ignored)]

    # Split into estimate and evaluate outputs -- use operate runs for evaluate outputs
    out_ds = out_full.filter(regex='^.*--get_ds$', axis=0).sort_index()
    out_op = out_full.filter(regex='^.*--get_op$', axis=0).sort_index()
    out_ds.index = ['--'.join(i.split('--')[:-1]) for i in out_ds.index]
    out_op.index = ['--'.join(i.split('--')[:-1]) for i in out_op.index]
    out_ds_benchmark = out_ds.filter(regex='^benchmark--.*$', axis=0)
    out_op_benchmark = out_op.filter(regex='^benchmark--.*$', axis=0)
    assert (out_ds.index == out_op.index).all()
    assert (out_ds.columns == out_op.columns).all()
    assert (out_ds_benchmark.index == out_op_benchmark.index).all()
    assert (out_ds_benchmark.columns == out_op_benchmark.columns).all()

    # Conduct some checks -- currently only implemented with totals
    if drop_regional:
        df_isclose = _isclose(out_ds, out_op)
        df_isclose_benchmark = df_isclose.filter(regex='^benchmark--.*$', axis=0)
        unmet_columns = ['peak_unmet_systemwide', 'gen_unmet_total']
        assert df_isclose.filter(regex='^cap_.*$', axis=1).all().all()
        assert df_isclose['sum_ts_weights'].all()
        assert df_isclose_benchmark['num_ts'].all()
        # Above doesn't work with wind as it has basically zero price, so no penalty for extra use
        assert (out_ds_benchmark['num_ts'] == out_ds_benchmark['sum_ts_weights']).all()
        assert (out_op_benchmark['num_ts'] == out_op_benchmark['sum_ts_weights']).all()
        assert np.allclose(out_ds[unmet_columns], 0., atol=0.1)
        assert np.allclose(
            out_ds.filter(regex='^agg_str_inp--no_sto--.*', axis=0)['cost_total'],
            out_ds.filter(regex='^agg_str_op_vars--no_sto--.*', axis=0)['cost_total']
        )  # For no storage model, operate variables should make no difference
        for replication in range(40):
            if replication in sims_ignored:
                continue
            out_ds_replication = out_ds.loc[out_ds['replication'] == replication]
            out_op_replication = out_op.loc[out_ds['replication'] == replication]
            assert out_ds_replication['sum_ts_weights'].nunique() == 1
            assert out_op_replication['sum_ts_weights'].nunique() == 1
            assert out_op_replication['demand_total'].round(0).nunique() == 1
    else:
        print('WARNING: Running without tests on outputs')

    out_ds['cluster_num_days'] = [_sim_id_to_num_days(i)[0] for i in out_ds.index]
    out_ds['cluster_num_days_extreme'] = [_sim_id_to_num_days(i)[1] for i in out_ds.index]
    out_op['cluster_num_days'] = [_sim_id_to_num_days(i)[0] for i in out_op.index]
    out_op['cluster_num_days_extreme'] = [_sim_id_to_num_days(i)[1] for i in out_op.index]

    return out_ds, out_op


def get_solution_time_info():
    '''Get info on solution times. Doesn't actually produce a plot but prints info to screen.'''

    experiments = {
        'example': {
            'has_benchmark': False,
            'num_years_base': 30,
            'agg_num_days_list': [120],
            'NUM_SIMULATIONS': 40  # In case not all have finished yet -- set to 40 when done
        },
        'validation': {
            'has_benchmark': True,
            'num_years_base': 3,
            'agg_num_days_list': [30, 120],
            'NUM_SIMULATIONS': 40  # In case not all have finished yet -- set to 40 when done
        }
    }
    a_priori_steps = ['agg_inp_mean']  # [design_A0]
    a_posteriori_steps = ['agg_inp_closest', 'agg_str_gencost_op_vars']  # [design_A0, design_A1]
    agg_methods = a_priori_steps + a_posteriori_steps

    for experiment, experiment_props in experiments.items():
        print(f'\nData for {experiment}:\n')
        NUM_SIMULATIONS = experiment_props['NUM_SIMULATIONS']
        has_benchmark = experiment_props['has_benchmark']
        num_years_base = experiment_props['num_years_base']
        agg_num_days_list = experiment_props['agg_num_days_list']

        # Read outputs and slice into model types, agg methods and number of representative days
        outputs_plan, outputs_operate = _read_outputs(num_years_base=num_years_base)

        # Function to get quantiles of solution times
        def get_solution_quantiles(outputs: pd.DataFrame, name: str, regex: str):
            '''Get quantiles of solution times, in minutes.'''
            solution_time_i = pd.Series(name=name, dtype='float')
            st = outputs.filter(regex=regex, axis=0)['solution_time']
            assert st.shape[0] == NUM_SIMULATIONS
            solution_time_i.loc['mean'] = st.mean() / 60.
            solution_time_i.loc['0.025'] = st.quantile(0.025) / 60.
            solution_time_i.loc['0.975'] = st.quantile(0.975) / 60.
            return solution_time_i

        # Function to get DataFrame of quantiles across aggregation methods
        def get_solution_info(outputs: pd.DataFrame):
            solution_times_list = []
            # Add benchmarks
            if has_benchmark:  # From outside function scope
                name = f'benchmark'
                regex = f'^benchmark--.*$'
                solution_times_list.append(
                    get_solution_quantiles(outputs=outputs, name=name, regex=regex)
                )
            # Add aggregated solutions
            for agg_method in agg_methods:
                for agg_num_days in agg_num_days_list:
                    name = f'{agg_method}--{agg_num_days}'
                    regex = f'^{agg_method}--..y--{agg_num_days:04d}d.*$'
                    solution_times_list.append(
                        get_solution_quantiles(outputs=outputs, name=name, regex=regex)
                    )
            solution_times = pd.concat(solution_times_list, axis=1).T
            return solution_times

        # Get solution time info, in minutes
        solution_info_plan = get_solution_info(outputs=outputs_plan)
        solution_info_operate = get_solution_info(outputs=outputs_operate)

        # Get total solution times for benchmark runs
        if experiment == 'validation':
            solution_info_benchmark = (
                solution_info_plan.loc[solution_info_plan.index.str.contains('benchmark')]
            )
            solution_info_benchmark.index = ['benchmark']

        # Get total solution times for a priori method -- just plan A0
        solution_info_a_priori = (
            solution_info_plan.loc[solution_info_plan.index.str.contains(a_priori_steps[0])]
        )
        solution_info_a_priori.index = [f'{agg_num_days}d' for agg_num_days in agg_num_days_list]

        # Get total solution times for a posteriori method -- plan A0, operate, plan A1 runs
        outputs_A0 = outputs_plan.filter(regex=f'^{a_posteriori_steps[0]}--.*$', axis=0)
        outputs_OP = outputs_operate.filter(regex=f'^{a_posteriori_steps[0]}--.*$', axis=0)
        outputs_A1 = outputs_plan.filter(regex=f'^{a_posteriori_steps[1]}--.*$', axis=0)
        match_cols = ['replication', 'cluster_num_days']
        assert np.allclose(outputs_A0[match_cols].values, outputs_OP[match_cols].values)
        assert np.allclose(outputs_A0[match_cols].values, outputs_A1[match_cols].values)
        solution_info_list = []
        for agg_num_days in agg_num_days_list:
            regex_A0 = f'^{a_posteriori_steps[0]}--..y--{agg_num_days:04d}d.*$'
            regex_A1 = f'^{a_posteriori_steps[1]}--..y--{agg_num_days:04d}d.*$'
            sol_time_A0 = outputs_plan.filter(regex=regex_A0, axis=0)['solution_time']
            sol_time_OP = outputs_operate.filter(regex=regex_A0, axis=0)['solution_time']
            sol_time_A1 = outputs_plan.filter(regex=regex_A1, axis=0)['solution_time']
            sol_time_total = pd.Series(
                sol_time_A0.to_numpy() + sol_time_OP.to_numpy() + sol_time_A1.to_numpy(),
                index=['--'.join(i.split('--')[1:]) for i in sol_time_A0.index]
            )
            info_i = pd.DataFrame()
            info_i.loc[f'{agg_num_days}d_TOTAL', 'mean'] = sol_time_total.mean() / 60.
            info_i.loc[f'{agg_num_days}d_TOTAL', '0.025'] = sol_time_total.quantile(0.025) / 60.
            info_i.loc[f'{agg_num_days}d_TOTAL', '0.975'] = sol_time_total.quantile(0.975) / 60.
            info_i.loc[f'{agg_num_days}d_A0', 'mean'] = sol_time_A0.mean() / 60.
            info_i.loc[f'{agg_num_days}d_A0', '0.025'] = sol_time_A0.quantile(0.025) / 60.
            info_i.loc[f'{agg_num_days}d_A0', '0.975'] = sol_time_A0.quantile(0.975) / 60.
            info_i.loc[f'{agg_num_days}d_OP', 'mean'] = sol_time_OP.mean() / 60.
            info_i.loc[f'{agg_num_days}d_OP', '0.025'] = sol_time_OP.quantile(0.025) / 60.
            info_i.loc[f'{agg_num_days}d_OP', '0.975'] = sol_time_OP.quantile(0.975) / 60.
            info_i.loc[f'{agg_num_days}d_A1', 'mean'] = sol_time_A1.mean() / 60.
            info_i.loc[f'{agg_num_days}d_A1', '0.025'] = sol_time_A1.quantile(0.025) / 60.
            info_i.loc[f'{agg_num_days}d_A1', '0.975'] = sol_time_A1.quantile(0.975) / 60.
            solution_info_list.append(info_i)
        solution_info_a_posteriori = pd.concat(solution_info_list, axis=0)

        if experiment == 'validation':
            print('Solution info, benchmark [mins, rounded]:\n')
            print(solution_info_benchmark.round(0).astype('int'), '\n')
        print('Solution info, a priori [mins, rounded]:\n')
        print(solution_info_a_priori.round(0).astype('int'), '\n')
        print('Solution info, a posteriori [mins, rounded]:\n')
        print(solution_info_a_posteriori.round(0).astype('int'), '\n\n\n\n')


def make_example_plot():
    '''Plot unserved energy levels across example simulation.'''

    # Parameters
    model_type_list = ['sto_m']
    num_years_base = 30
    agg_num_days_list = [120]
    reduction_method_list = ['agg_inp_mean', 'agg_str_gencost_op_vars']

    # Read outputs and slice into those required for plot
    reduction_method_list_regex = f'({"|".join(reduction_method_list)})'
    outputs = (
        _read_outputs(num_years_base=num_years_base, drop_regional=True)[1]
        .filter(regex=reduction_method_list_regex, axis=0)
    )

    # Plot distribution of error metrics for each model and subsample size across subsample methods
    # Plots: cap_baseload, cap_peaking, cap_wind, cap_storage, gen_unmet
    for agg_num_days in agg_num_days_list:
        fig, ax = plt.subplots(figsize=(3.5, 4))

        # Slice results into hose for this model type and number of representative days
        regex = f'^.*--{num_years_base:02d}y--{agg_num_days:04d}d_...dh--....$'
        outputs_plot = outputs.filter(regex=regex, axis=0)

        # Get quantiles of energy unserved for each method
        quantiles_list = []
        for method in reduction_method_list:
            outputs_i = outputs_plot.copy().filter(regex=f'^{method}--.*', axis=0)
            outputs_i['gen_unmet_total_proportion'] = np.divide(
                outputs_i['gen_unmet_total'].to_numpy(), outputs_i['demand_total'].to_numpy()
            )
            # Calculate quantiles and add to quantiles_list
            quantiles_i = outputs_i.quantile(q=[0.025, 0.25, 0.5, 0.75, 0.975])
            quantiles_i.index = pd.MultiIndex.from_product([[method], quantiles_i.index])
            quantiles_list.append(quantiles_i)
        quantiles = pd.concat(quantiles_list)[['gen_unmet_total_proportion']]

        # Plot quantiles of energy unserved
        _plot_metrics_fixed_subsample_size_across_methods(
            ax=ax,
            quantiles=quantiles,
            reduction_method_list=reduction_method_list,
            col_name='gen_unmet_total_proportion',
            ylabel='Energy unserved [%]',
            ylim=[-0.005, 0.2]
        )

        title = f'{num_years_base} years aggregated to\n{agg_num_days} representative days'
        ax.set_xticklabels(['A', 'F'])
        labels = [
            'A (a priori) :  Representative = cluster mean',
            ('                      Include days with high\n'
                'F (a post.)  : $\:$ generation cost, cluster\n'
                '                      using storage decisions')
        ]
        fig.legend(
            handles=_get_legend_handles(ax=ax, labels=labels),
            loc='lower center', ncol=1, bbox_to_anchor=(0.5, 0.0), fontsize=9
        )
        ax.set_title(title)
        fig.tight_layout()
        plt.subplots_adjust(left=0.18, bottom=0.3)
        save_filename = f'example.pdf'
        plt.savefig(f'{config.OUTPUTS_DIR}/plots_post/{save_filename}')
        plt.close()


def make_evaluation_plots():
    '''Plot metrics for evaluation exercise with different aggregation methods.'''

    # Parameters
    num_years_base = 3
    agg_num_days_list = [30, 120]
    agg_method_list = [
        'agg_inp_mean',
        'agg_inp_closest',
        'agg_inp_min_max',
        'agg_str_unmet_inp',
        'agg_str_gencost_inp',
        'agg_str_gencost_op_vars'
    ]
    ylims = {
        'cap_baseload': {30: [-110, 110], 60: [-110, 110], 120: [-110, 110]},
        'cap_peaking': {30: [-110, 110], 60: [-110, 110], 120: [-110, 110]},
        'cap_wind': {30: [-110, 110], 60: [-110, 110], 120: [-110, 110]},
        'cap_transmission': {30: [-25, 25], 60: [-25, 25], 120: [-25, 25]},
        'cap_storage': {30: [-110, 40], 60: [-110, 40], 120: [-110, 40]},
        'gen_unmet': {30: [-0.16, 1.6], 60: [-0.05, 0.5], 120: [-0.032, 0.32]}
    }

    # Read outputs and slice into those required for plot
    agg_method_list_regex = f'(benchmark|{"|".join(agg_method_list)})'
    outputs = (
        _read_outputs(num_years_base=num_years_base, drop_regional=True)[1]
        .filter(regex=agg_method_list_regex, axis=0)
    )

    # Plot distribution of error metrics for each model and subsample size across subsample methods
    # Plots: cap_baseload, cap_peaking, cap_wind, cap_storage, gen_unmet
    outputs_benchmark = outputs.filter(regex=f'^benchmark--.*$', axis=0)
    for agg_num_days in agg_num_days_list:
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(7, 5))

        # Slice results into hose for this model type and number of representative days
        regex = f'^.*--{num_years_base:02d}y--{agg_num_days:04d}d_...dh--....$'
        outputs_plot = outputs.filter(regex=regex, axis=0)

        # Get quantiles for each method
        quantiles_list = []
        for method in agg_method_list:
            outputs_i = outputs_plot.copy().filter(regex=f'^{method}--.*', axis=0)
            # Calculate deviations from benchmark
            numerator = np.subtract(
                outputs_i[DEVIATION_COLUMNS].clip(lower=ZERO_CLIP).to_numpy(),
                outputs_benchmark[DEVIATION_COLUMNS].clip(lower=ZERO_CLIP).to_numpy()
            )
            denominator = (
                outputs_benchmark[DEVIATION_COLUMNS].clip(lower=ZERO_CLIP).to_numpy()
            )
            deviations_np = np.divide(numerator, denominator)
            outputs_i[[f'deviation_{i}' for i in DEVIATION_COLUMNS]] = deviations_np
            outputs_i['gen_unmet_total_proportion'] = np.divide(
                outputs_i['gen_unmet_total'].to_numpy(),
                outputs_benchmark['demand_total'].to_numpy()
            )
            # Calculate quantiles and add to quantiles_list
            quantiles_i = outputs_i.quantile(q=[0.025, 0.25, 0.5, 0.75, 0.975])
            quantiles_i.index = pd.MultiIndex.from_product([[method], quantiles_i.index])
            quantiles_list.append(quantiles_i)
        quantiles = pd.concat(quantiles_list)

        _plot_metrics_fixed_subsample_size_across_methods(
            ax=axes[0][0],
            quantiles=quantiles,
            reduction_method_list=agg_method_list,
            col_name='deviation_cap_baseload_total',
            title='Baseload error [% of MW]',
            ylim=ylims['cap_baseload'][agg_num_days]
        )
        _plot_metrics_fixed_subsample_size_across_methods(
            ax=axes[0][1],
            quantiles=quantiles,
            reduction_method_list=agg_method_list,
            col_name='deviation_cap_peaking_total',
            title='Peaking error [% of MW]',
            ylim=ylims['cap_peaking'][agg_num_days]
        )
        _plot_metrics_fixed_subsample_size_across_methods(
            ax=axes[0][2],
            quantiles=quantiles,
            reduction_method_list=agg_method_list,
            col_name='deviation_cap_wind_total',
            title='Wind error [% of MW]',
            ylim=ylims['cap_wind'][agg_num_days]
        )
        _plot_metrics_fixed_subsample_size_across_methods(
            ax=axes[1][0],
            quantiles=quantiles,
            reduction_method_list=agg_method_list,
            col_name='deviation_cap_transmission_total',
            title='Transmission error [% of MW]',
            ylim=ylims['cap_transmission'][agg_num_days]
        )
        _plot_metrics_fixed_subsample_size_across_methods(
            ax=axes[1][1],
            quantiles=quantiles,
            reduction_method_list=agg_method_list,
            col_name='deviation_cap_storage_energy_total',
            title='Storage error [% of MWh]',
            ylim=ylims['cap_storage'][agg_num_days]
        )
        _plot_metrics_fixed_subsample_size_across_methods(
            ax=axes[1][2],
            quantiles=quantiles,
            reduction_method_list=agg_method_list,
            col_name='gen_unmet_total_proportion',
            title='Energy unserved [% of MWh]     ',
            ylim=ylims['gen_unmet'][agg_num_days]
        )

        labels = ['A', 'B', 'C', 'D', 'E', 'F'][:len(agg_method_list)]
        for ax in [axes[0][0], axes[0][1], axes[0][2]]:
            ax.set_xticklabels([])
        for ax in [axes[1][0], axes[1][1], axes[1][2]]:
            ax.set_xticklabels(labels)

        # Add legend
        labels = [
            'A (a priori) : Representative = cluster mean',
            'B (a priori) : Representative = closest day (medoid)',
            'C (a priori) : Include max demand + min wind day',
            'D (a post.)  : Include days with energy unserved',
            ('E (a post.)  : Include days with high generation cost, '
                'cluster using time series inputs'),
            ('F (a post.)  : Include days with high generation cost, '
                'cluster using storage decisions')
        ]
        assert len(labels) == len(agg_method_list)
        fig.legend(
            handles=_get_legend_handles(ax=axes[0][0], labels=labels),
            loc='lower center', ncol=1, bbox_to_anchor=(0.53, 0.0), fontsize=9
        )

        fig.suptitle(f'3 years aggregated to {agg_num_days} representative days', fontsize=12)
        fig.tight_layout()
        plt.subplots_adjust(hspace=0.3, wspace=0.4, bottom=0.35)
        save_filename = f'evaluation_{agg_num_days}d.pdf'
        plt.savefig(f'{config.OUTPUTS_DIR}/plots_post/{save_filename}')
        plt.close()


def main():
    get_solution_time_info()
    make_example_plot()
    make_evaluation_plots()


if __name__ == '__main__':
    main()
