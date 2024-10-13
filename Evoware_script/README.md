# **Evoware Script for Tecan Liquid Handler Control**

This folder contains scripts used to execute commands for controlling the **Tecan liquid handler** through the **Evoware** software. It includes subroutines for interfacing with the liquid handler and optimization scripts for both low and high dimensional experimental designs.

---

## **1. Converting Interfacing Scripts to .exe**

### 1.1 Modify the Path in Python Scripts
Before running the Evoware interfacing scripts, you need to modify the file paths in the following scripts:

- **Navigate to:**
    \Evoware_script\subroutines\

- **Files to modify:**
- `tecan_watching.py`: Change the paths as needed.
- `move_recan_rawdata.py`: Update the file paths in this script as well.

### 1.2 Convert Scripts to Executables

- Convert the modified Python scripts to **`.exe`** files.
- Move the generated **`.exe`** files to the following folder:
\dist


---

## **2. Optimization Scripts for Discovery and Heat Challenge**

### **Low-Dimensional Optimization (n < 12)**

The script used for **low-dimensional optimization** (where the number of parameters is less than 12) is located at:

    \Evoware_script\formal_script\Closed_loop_discovery_heat_challenge.esc


### **High-Dimensional Optimization (n ≥ 12)**

For **high-dimensional optimization**, the following script is used:

\Evoware_script\formal_script\Closed_loop_discovery_heat_challenge.esc


### **Before Running the Scripts:**

Make sure to update the following paths:
- **`nas_project_path`**: Set this to the correct project path on your NAS (Network Attached Storage).
- **`local_subroutine_path`**: Ensure this is set to the appropriate local path where subroutines are stored.



