import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors


plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams["mathtext.rm"] = "serif"
plt.rcParams["mathtext.bf"] = "serif:bold"
plt.rcParams["mathtext.default"] = "regular"


COLOR_GREY = colors.ColorConverter.to_rgba('grey', alpha=0.5)
COLOR_WIND = colors.ColorConverter.to_rgba('green', alpha=0.5)
COLOR_SOLAR = colors.ColorConverter.to_rgba('gold', alpha=0.5)
COLOR_BASELOAD = colors.ColorConverter.to_rgba('royalblue', alpha=0.8)
COLOR_PEAKING = colors.ColorConverter.to_rgba('cornflowerblue', alpha=0.6)
COLOR_STORAGE = colors.ColorConverter.to_rgba('black', alpha=0.0)  # Clear
HATCH_STORAGE = '....'
COLOR_UNMET = colors.ColorConverter.to_rgba('red', alpha=0.5)

EDGE_GREY = colors.ColorConverter.to_rgba('grey', alpha=1)
EDGE_WIND = colors.ColorConverter.to_rgba('green', alpha=1)
EDGE_SOLAR = colors.ColorConverter.to_rgba('orange', alpha=1)
EDGE_BASELOAD = colors.ColorConverter.to_rgba('royalblue', alpha=1)
EDGE_PEAKING = colors.ColorConverter.to_rgba('cornflowerblue', alpha=1)
EDGE_STORAGE = colors.ColorConverter.to_rgba('black', alpha=0.5)
EDGE_UNMET = colors.ColorConverter.to_rgba('red', alpha=1)


def _plot_generation_levels(
    ax,
    ts_out: pd.DataFrame,
    ts_orig: pd.DataFrame = None,
    ylim_gen: list = None,
    vlines: list = None,
    title: str = None,
    annotate_text: str = None
):
    x = ts_out.index
    demand = ts_out['demand_total']
    demand_orig = ts_orig['demand_total'] if ts_orig is not None else None  # Demand pre-aggregation

    # Get cumulate generation levels by stacking generation levels
    techs_in_stack_order = ['baseload', 'peaking', 'wind', 'unmet', 'storage']
    stack_levels = {}
    y1 = 0
    level_num = 0
    for tech in techs_in_stack_order:
        for region in range(1, 7):
            gen_tech_region = f'gen_{tech}_region{region}'
            if gen_tech_region in ts_out.columns:
                y2 = y1 + ts_out[gen_tech_region]
                fill_color = globals()[f'COLOR_{tech.upper()}']
                edge_color = globals()[f'EDGE_{tech.upper()}']
                hatch = HATCH_STORAGE if tech == 'storage' else None
                stack_levels[f'level_{level_num}'] = {
                    'y1': y1, 'y2': y2, 'label': tech,
                    'fill_color': fill_color, 'edge_color': edge_color, 'hatch': hatch
                }
                y1 = y2
                level_num += 1

    # Plot shaded regions
    for level_num in range(len(stack_levels)):
        level_info = stack_levels[f'level_{level_num}']
        label = level_info['label'] if level_num % 3 == 0 else None
        ax.fill_between(
            x=x, y1=level_info['y1'], y2=level_info['y2'], label=label,
            color=level_info['fill_color'], hatch=level_info['hatch'], linewidth=0
        )

    # Plot lines -- do this in reverse so the "lowest" technology is shown on top
    for level_num in reversed(range(len(stack_levels))):
        level_info = stack_levels[f'level_{level_num}']
        ax.plot(x, level_info['y2'], color=level_info['edge_color'], linewidth=2)
    if demand_orig is not None:
        ax.plot(x, demand_orig, label='orig demand', color='black', linewidth=1, alpha=0.5)
    ax.plot(x, demand, label='agg demand', color='black', linewidth=2)

    # Plot vertical lines
    if vlines is not None:
        for vline in vlines:
            ax.axvline(x=vline, color='black', linestyle=':')

    # Cosmetic changes
    # for tick in ax.get_xticklabels():
    #     tick.set_rotation(30)
    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    xlim = [x[0], x[-1]]
    ax.set_xlim(xlim)
    ax.set_ylim(bottom=0)
    if ylim_gen is not None:
        ax.set_ylim(ylim_gen)
    ax.set_ylabel('Generation (GW)')

    if title is not None:
        ax.set_title(title)

    # Annotate the text
    if annotate_text is not None:
        x_offset, y_offset = 0.01, -0.02  # Offset from top left
        x_text = xlim[0] + x_offset * (xlim[-1] - xlim[0])
        y_text = ax.get_ylim()[1] + y_offset * (ax.get_ylim()[1] - ax.get_ylim()[0])
        ax.text(x=x_text, y=y_text, s=annotate_text, ha='left', va='top')


def _plot_storage_levels(
    ax,
    ts_out: pd.DataFrame,
    cap_storage_energy: int = None,
    vlines: list = None,
    annotate_text: str = None
):
    x = ts_out.index
    ts_out.columns = [i.replace('_total', '') for i in ts_out.columns]  # Remove '_total' suffix
    level_storage = ts_out['level_storage']
    ax.fill_between(x, 0, level_storage, color=COLOR_PEAKING)
    ax.plot(x, level_storage, color=EDGE_PEAKING, linewidth=2)

    # Plot vertical lines
    if vlines is not None:
        for vline in vlines:
            ax.axvline(x=vline, color='black', linestyle=':')

    ax.axhline(y=0, color='black', linestyle=':')
    if cap_storage_energy is not None:
        ax.axhline(y=cap_storage_energy, color='black', linestyle=':')
    ax.set_ylabel('Storage level (GWh)')
    xlim = [x[0], x[-1]]
    ax.set_xlim(xlim)

    # Annotate the text
    if annotate_text is not None:
        x_offset, y_offset = 0.01, -0.05  # Offset from top left
        x_text = xlim[0] + x_offset * (xlim[-1] - xlim[0])
        y_text = ax.get_ylim()[1] + y_offset * (ax.get_ylim()[1] - ax.get_ylim()[0])
        ax.text(x=x_text, y=y_text, s=annotate_text, ha='left', va='top')


def plot_model_timeseries(
    ts_out_hourly: pd.DataFrame,
    ts_out_daily: pd.DataFrame,
    ts_orig_hourly: pd.DataFrame = None,
    ts_orig_daily: pd.DataFrame = None,
    open_plot: bool = False,
    plot_save_filepath: str = None,
    annotate_text: str = None,
    cap_storage_energy: float = None,
    ylim_gen: list = None,
):
    '''Make a plot of generation, storage (dis)charge levels and storage levels.'''

    fig, axes = plt.subplots(
        nrows=4, figsize=(10, 12), gridspec_kw={'height_ratios': [2, 1, 2, 1]}, sharex=True
    )
    vlines = [ts_out_hourly.index[0], ts_out_hourly.index[-1]]  # Plot x limits of hourly plot

    # Calculate systemwide totals
    for ts in [ts_out_hourly, ts_out_daily, ts_orig_hourly, ts_orig_daily]:
        columns_to_sum = [
            'demand', 'gen_baseload', 'gen_peaking', 'gen_wind', 'gen_unmet',
            'level_storage_intraday', 'level_storage_interday', 'level_storage'
        ]
        for column in columns_to_sum:
            ts[f'{column}_total'] = ts.filter(regex=f'^{column}_region.$').sum(axis=1)

    _plot_generation_levels(
        ax=axes[0],
        ts_out=ts_out_hourly,
        ts_orig=ts_orig_hourly,
        ylim_gen=ylim_gen,
        title='Hourly'
    )
    _plot_storage_levels(
        ax=axes[1],
        ts_out=ts_out_hourly,
        cap_storage_energy=cap_storage_energy
    )
    _plot_generation_levels(
        ax=axes[2],
        ts_out=ts_out_daily,
        ts_orig=ts_orig_daily,
        ylim_gen=ylim_gen,
        vlines=vlines,  # Plot x limits of hourly plot on daily plot
        title='Daily'
    )
    _plot_storage_levels(
        ax=axes[3],
        ts_out=ts_out_daily,
        vlines=vlines,  # Plot x limits of hourly plot on daily plot
        annotate_text=annotate_text,
        cap_storage_energy=cap_storage_energy
    )
    axes[3].set_xlim([ts_out_hourly.index[0], ts_out_hourly.index[-1] + pd.Timedelta(1, unit='h')])

    handles, labels = axes[2].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=8, bbox_to_anchor=(0.5, 0.22))
    fig.tight_layout()
    # fig.subplots_adjust(bottom=0.23)

    # Save and open the plot if requested
    if plot_save_filepath is not None:
        plt.savefig(plot_save_filepath)
    if open_plot:
        plt.show()

    plt.close()
