# main_example.sh : Simulations for 'example' experiment
# ------------------------------------------------------
#
# Notes:
# - Master model config, shared across all simulations, defined in `config.py`


# Parameters
# ----------
EXTRA_CONFIG_NAME="storage_medium_cost"
TS_BASE_RESAMPLE_NUM_YEARS=1
TS_REDUCTION_NUM_DAYS=10
SIMULATION_NAME_LIST=(
    "agg_inp_mean"
    # "agg_inp_closest"
    # "agg_str_gencost_op_vars"
)
SIMULATION_TYPE_LIST=("get_design_estimate" "get_operate_variables")


# Simulations
# -----------
for SIMULATION_NAME in "${SIMULATION_NAME_LIST[@]}"
do
    for SIMULATION_TYPE in "${SIMULATION_TYPE_LIST[@]}"
    do
        for REPLICATION in {0..0}
        do
            python3 main.py \
                    --simulation_name $SIMULATION_NAME \
                    --simulation_type $SIMULATION_TYPE \
                    --extra_config_name $EXTRA_CONFIG_NAME \
                    --ts_base_resample_num_years $TS_BASE_RESAMPLE_NUM_YEARS \
                    --ts_reduction_num_days $TS_REDUCTION_NUM_DAYS \
                    --replication $REPLICATION
            printf "\n\n"
        done
    done
done


# sh move_data.sh collate  # Collate summary outputs into single file


