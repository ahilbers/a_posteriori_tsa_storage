import os


# # Constants when run in parallel in compute clusters (HPC = high performace computing)
# # In HPC parallel run, PRN (parallel run number) is index corresponding to parallel run. Else set
# # PRN=0 and code is run in sequence. On laptop, environment variables are set in `main.sh`.
# PRN = int(os.environ['PRN']) - 1  # Offset by 1 to make ranges start at 0
# RUN_ID = f'{os.environ["PBS_JOBNAME"]}_{PRN:03d}'
# REPLICATIONS_SUPERSET = range(0, 100)  # Should cover all iterations, most of these are not run


TS_MASTER_FIRST_YEAR = 1980  # First year in master time series CSV (before any slicing/subsampling
OUTPUTS_DIR = 'outputs'
LOG_SAVE_DIR = f'{OUTPUTS_DIR}/logs'
RUN_CONFIG_SAVE_DIR = f'{OUTPUTS_DIR}/configs'
SUMMARY_OUTPUTS_SAVE_DIR = f'{OUTPUTS_DIR}/summary_outputs'
TS_OUTPUTS_SAVE_DIR = f'{OUTPUTS_DIR}/ts_outputs'
PLOT_SAVE_DIR = f'{OUTPUTS_DIR}/plots'
NUMBER_OF_CAPACITIES = 22  # Number of generation, transmission, storage capacities in system design


# Arguments specified when calling 'main.py', usually from bash script
COMMAND_LINE_ARGUMENTS = {
    'simulation_name': {
        'required': True, 'default': None, 'type': str
    },
    'simulation_type': {
        'required': True, 'default': None, 'type': str
    },
    'ts_base_resample_num_years': {
        'required': False, 'default': None, 'type': int
    },
    'ts_reduction_num_days': {
        'required': False, 'default': None, 'type': int
    },
    'replication': {
        'required': True, 'default': None, 'type': int
    },
}


# Base config used for each model run -- many values for specific runs are overwritten later in
# pipeline, e.g. via command line arguments, calculations or additional configs defined below
main_run_config = {
    'simulation': {  # All these values are overwritten in `main.py`
        'name': None,
        'type': None,
        'id': None,
        'replication': None,
    },
    'model': {
        'model_name': '6_region',
        'run_mode': 'plan',  # Operate models: 'plan' mode with fixed capacities -- same effect
        'baseload_integer': False,
        'baseload_ramping': False,
        'allow_unmet': False,
        'fixed_caps': {},  # Can be overwritten during simulations
        'extra_override_name': None,
        'extra_override_dict': None,
    },
    'ts_base': {
        'column_subset': None,
        'time_slice': None,
        # Randomly resample years for new base time series, set using command line argument
        'resample_num_years': None,
        'resample_years_list': None,  # Length should match 'resample_num_years'
        'roll_days': 184,  # Roll last N days to front -- reduce impact of intial storage level
    },
    'ts_aggregation': {  # Many values get overwritten by extra configs, defined below
        # Any columns in 'column(s) used' can be either time series input or operate variable
        'aggregate': False,
        'num_days': None,  # Number of representative days
        'stratification': {
            'stratify': False,  # Stratify into 'extreme' and 'regular' days before clustering
            'column_used': 'generation_cost',  # Column used to stratify -- determine 'extreme' days
            'aggfunc': 'sum',  # Use daily total ('sum') or maximum ('max')
            'ts_base_split_extreme': 0.05,  # Proportion of base time series considered 'extreme'
            'ts_agg_split_extreme': 0.5  # Proportion of representative days for 'extreme' region
        },
        'clustering': {
            'columns_used': [
                'demand_region2', 'demand_region4', 'demand_region5',
                'wind_region2', 'wind_region5', 'wind_region6',
            ],  # Sometimes modified to add storage (dis)charge decisions also (see below)
            'normalize_method': 'z-transform',  # None, 'z-transform' or 'min-max'
        },
        'representative_day': 'closest'  # 'mean' or 'closest' (medoid)
    },
    'save': {
        'log_filepath': f'{LOG_SAVE_DIR}/main.log',
        'log_level_file': 'INFO',  # Level for .log file
        'log_level_stdout': 'INFO',  # Level for stdout (including terminal)
        'save_run_config': True,
        'save_summary_outputs': True,
        'save_ts_outputs': False,  # Gets changed to True for 'get_operate_variables' simulations
        'save_plot': False,
        'save_full_outputs': False,
        'plot_ts_slice': [3700, 4336],  # Index range to plot, .iloc format (e.g. [3700, 4336])
    }
}


# Specify settings here that overwrite defauls
agg_inp_mean = {  # Method A
    'ts_aggregation': {
        'aggregate': True,
        'representative_day': 'mean'
    }
}
agg_inp_closest = {  # Method B
    'ts_aggregation': {
        'aggregate': True,
        'representative_day': 'closest'
    }
}
agg_inp_min_max = {  # Method C
    'ts_aggregation': {
        'aggregate': True,
        'stratification': {
            'stratify': True,
            'column_used': 'max_demand_min_wind',  # Not actually a column
            'aggfunc': 'max',  # Get day with max hourly demand, min hourly wind
            # Max demand, min wind in 3 regions each -- max 6 in all, splits below not applicable
            'ts_base_split_extreme': None,
            'ts_agg_split_extreme': None
        }
    }
}
agg_str_unmet_inp = {  # Method D
    'ts_aggregation': {
        'aggregate': True,
        'stratification': {
            'stratify': True,
            'column_used': 'gen_unmet_total'
        }
    }
}
agg_str_gencost_inp = {  # Method E
    'ts_aggregation': {
        'aggregate': True,
        'stratification': {
            'stratify': True
        }
    }
}
agg_str_gencost_op_vars = {  # Method F
    'ts_aggregation': {
        'aggregate': True,
        'stratification': {
            'stratify': True
        },
        'clustering': {
            'columns_used': [
                'demand_region2', 'demand_region4', 'demand_region5',
                'wind_region2', 'wind_region5', 'wind_region6',
                'gen_storage_region2', 'gen_storage_region5', 'gen_storage_region6'
            ]
        }
    }
}
