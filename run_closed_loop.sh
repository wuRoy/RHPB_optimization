# read the 'mode' from project_config.ymal
mode=$(grep 'mode:' project_config.yaml | awk '{print $2}' | tr -d "'")
echo "Mode is: $mode"
nohup python -u src/run_loop/run_closed_loop.py > Experiments/out_$mode.log 2>&1
