# Simulate Imperial HPC batch job in laptop

export PBS_JOBNAME="laptop"
PBS_ARRAY_INDEX_FIRST=1
PBS_ARRAY_INDEX_LAST=1

for PBS_ARRAY_INDEX_VALUE in $(seq $PBS_ARRAY_INDEX_FIRST $PBS_ARRAY_INDEX_LAST)
do
    # If PBS_ARRAY_INDEX = 0, then run all simulations inside python in one go
    export PBS_ARRAY_INDEX=$PBS_ARRAY_INDEX_VALUE
    sh main_validation.sh
    printf "\n\n\n\n"
done


# Clean up outputs directory
# python3 organise_outputs.py collate filename=summary_outputs.csv
# python3 organise_outputs.py clean