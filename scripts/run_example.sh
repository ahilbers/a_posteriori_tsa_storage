# main_example.sh : Simulations for 'example' experiment
# ------------------------------------------------------
#
# Notes:
# - Master model config, shared across all simulations, defined in `config.py`


# Parameters
# ----------
SIMULATION_NAME_LIST=(
    "agg_inp_mean"  # Method A
    "agg_inp_closest"  # Method B: (a priori) first stage of method F
    "agg_str_gencost_op_vars"  # Method F: (a posteriori) second stage
)
SIMULATION_TYPE_LIST=("get_design_estimate" "get_operate_variables")
TS_BASE_RESAMPLE_NUM_YEARS=30
TS_REDUCTION_NUM_DAYS=120


# Simulations
# -----------
for SIMULATION_NAME in "${SIMULATION_NAME_LIST[@]}"
do
    for SIMULATION_TYPE in "${SIMULATION_TYPE_LIST[@]}"
    do
        for REPLICATION in {0..39}
        do
            python3 main.py \
                    --simulation_name $SIMULATION_NAME \
                    --simulation_type $SIMULATION_TYPE \
                    --ts_base_resample_num_years $TS_BASE_RESAMPLE_NUM_YEARS \
                    --ts_reduction_num_days $TS_REDUCTION_NUM_DAYS \
                    --replication $REPLICATION
            printf "\n\n"
        done
    done
done


