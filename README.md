# RHP Blend Optimization

This project optimizes the composition of RHP blends through iterative closed-loop experiments. Follow the steps below to set up the environment, configure parameters, and run the optimization process.

---

## Setup and Installation

1. **Install the Conda Environment**  
    Use the provided `environment.yml` file to install the required environment:
    ```bash
    conda env create -f environment.yml

2. **Install the Project in Development Mode**  
    Install the package locally with:
    ```bash
    python setup.py develop

3. **Specify the Configurations**  
    Specify the necessary parameters in project_config.yaml to fit the optimization requirements.

4. **Run the Optimization Loop**  
    Execute the optimization process with:
    ```bash
    bash run_closed_loop.sh

5. **Move the Proposed Compositions to the Experiment Directory**  
    Move the generated proposed_composition.csv to the experiment data directory:
    ```bash
    ../Experiments/project_dir/data/round_1
    The proposed_composition.csv should contain a 96 × d matrix, where d is the number of RHPs to be blended.

6. **Specify a Customized Sequence**  
    To use a predefined sequence:
    Update the project_config.yaml to select predefined mode.
    Place your .csv files in the job_list directory.
    Run the loop:
    ```bash
    bash run_closed_loop.sh
    After running, move the .csv files to the experiment directory:
    ```bash
    bash run_closed_loop.sh


