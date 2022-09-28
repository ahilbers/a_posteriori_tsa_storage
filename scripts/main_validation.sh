# main_validation.sh : Simulations for 'validation' experiment
# ------------------------------------------------------------
#
# Notes:
# - Master model config, shared across all simulations, defined in `config.py`


# Parameters
# ----------
EXTRA_CONFIG_NAME_LIST=("storage_medium_cost")
TS_BASE_RESAMPLE_NUM_YEARS_LIST=(1)
TS_REDUCTION_NUM_DAYS_LIST=(30)
SIMULATION_NAME_LIST=(
    "benchmark"
    "agg_inp_mean"
    "agg_inp_closest"
    "agg_inp_min_max"
    "agg_str_unmet_inp"
    "agg_str_gencost_inp"
    "agg_str_gencost_op_vars"
)
SIMULATION_TYPE_LIST=("get_design_estimate")


# Simulations
# -----------
for REPEAT in {1..2}  # Repeat twice in case some jobs fail first time
do
    for SIMULATION_NAME in "${SIMULATION_NAME_LIST[@]}"
    do
        for TS_BASE_RESAMPLE_NUM_YEARS in "${TS_BASE_RESAMPLE_NUM_YEARS_LIST[@]}"
        do
            for EXTRA_CONFIG_NAME in "${EXTRA_CONFIG_NAME_LIST[@]}"
            do
                for SIMULATION_TYPE in "${SIMULATION_TYPE_LIST[@]}"
                do
                    for TS_REDUCTION_NUM_DAYS in "${TS_REDUCTION_NUM_DAYS_LIST[@]}"
                    do
                        python3 main.py \
                                --simulation_name $SIMULATION_NAME \
                                --simulation_type $SIMULATION_TYPE \
                                --extra_config_name $EXTRA_CONFIG_NAME \
                                --ts_base_resample_num_years $TS_BASE_RESAMPLE_NUM_YEARS \
                                --ts_reduction_num_days $TS_REDUCTION_NUM_DAYS
                        printf "\n\n"
                    done
                done
            done
        done
    done
done


sh move_data.sh collate  # Collate summary outputs into single file


