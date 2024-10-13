# RHP Blend Optimization

This project optimizes the composition of RHP blends through iterative closed-loop experiments. Follow the steps below to set up the environment, configure parameters, and run the optimization process.

---

## Setup and run the optimization

1. **Install the Conda Environment**  

    Install the environment:
    ```bash
    conda env create -f environment.yml
    ```
2. **Install the Project in Development Mode**  

    Install the package locally with:
    ```bash
    python setup.py develop
    ```
3. **Specify the Configurations**  

    Specify the necessary parameters in `project_config.yaml` for the optimization.

4. **Run the Optimization Loop**  

    Execute the optimization process with:
    ```bash
    bash run_closed_loop.sh
    ```

5. **Move the Proposed Compositions to the Experiment Directory**  

    Move the generated proposed_composition.csv to the experiment data directory:
    ```bash
    ../Experiments/project_name/data/round_1
    ```
    project_name is the name of the project you specified in the `project_config.yaml`.
    The `proposed_composition.csv` should contain a 96 × d matrix, where d is the number of RHPs to be blended.
    Once the `tecan_command.csv`is generated, you can start the autonomous experiments by running the Evoware program.

## Specify a Customized Sequence 

    Besides running a autonomous optimization campaign, you could also run a pre-defined sequence of jobs with the system.
    Specify the predefined mode in the `project_config.yaml`. Place your .csv files in a folder called `job_list`.
    Run the loop:

    ```bash
    bash run_closed_loop.sh  
    ```
    
    Move the `job_list` to the experiment directory `../Experiments/project_name` and then start the Evoware program.
