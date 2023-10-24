# main_validation.sh : Simulations for 'validation' experiment
# ------------------------------------------------------------
#
# Notes:
# - Master model config, shared across all simulations, defined in `config.py`


# Parameters
# ----------
SIMULATION_NAME_LIST=(
    "benchmark"
    "agg_inp_mean"  # Method A
    "agg_inp_closest"  # Method B
    "agg_inp_min_max"  # Method C
    "agg_str_unmet_inp"  # Method D
    "agg_str_gencost_inp"  # Method E
    "agg_str_gencost_op_vars"  # Method F
)
SIMULATION_TYPE_LIST=("get_design_estimate")
TS_BASE_RESAMPLE_NUM_YEARS=3


# Simulations, 30 representative days
# -----------------------------------
for SIMULATION_NAME in "${SIMULATION_NAME_LIST[@]}"
do
    for SIMULATION_TYPE in "${SIMULATION_TYPE_LIST[@]}"
    do
        for REPLICATION in {0..1}
        do
            python3 main.py \
                    --simulation_name $SIMULATION_NAME \
                    --simulation_type $SIMULATION_TYPE \
                    --ts_base_resample_num_years $TS_BASE_RESAMPLE_NUM_YEARS \
                    --ts_reduction_num_days 30 \
                    --replication $REPLICATION
            printf "\n\n"
        done
    done
done


# Simulations, 120 representative days
# ------------------------------------
for SIMULATION_NAME in "${SIMULATION_NAME_LIST[@]}"
do
    for SIMULATION_TYPE in "${SIMULATION_TYPE_LIST[@]}"
    do
        for REPLICATION in {0..1}
        do
            python3 main.py \
                    --simulation_name $SIMULATION_NAME \
                    --simulation_type $SIMULATION_TYPE \
                    --ts_base_resample_num_years $TS_BASE_RESAMPLE_NUM_YEARS \
                    --ts_reduction_num_days 120 \
                    --replication $REPLICATION
            printf "\n\n"
        done
    done
done


