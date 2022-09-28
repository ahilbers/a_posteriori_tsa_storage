# main_example.sh : Simulations for 'example' experiment
# ------------------------------------------------------
#
# Notes:
# - Master model config, shared across all simulations, defined in `config.py`


# Parameters
# ----------
EXTRA_CONFIG_NAME_LIST=("storage_none")
TS_BASE_RESAMPLE_NUM_YEARS_LIST=(10)
TS_REDUCTION_NUM_DAYS_LIST=(90)
SIMULATION_NAME_LIST=(
    "agg_inp_closest"
    "agg_op_vars"
)
SIMULATION_TYPE_LIST=("get_design_estimate" "get_operate_variables")


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


