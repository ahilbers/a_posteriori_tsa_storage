# Run all simulations and create plots

printf "Running example simulations.\n\n"
# sh scripts/run_example.sh
printf "Done running example simulations.\n\n\n\n\n\n"

printf "Running example simulations.\n\n"
# sh scripts/run_validation.sh
printf "Done running example simulations.\n\n\n\n\n\n"

printf "Creating plots.\n\n"
sh scripts/make_figures.sh
printf "Done.\n"