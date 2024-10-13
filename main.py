import subprocess
import os
import sys

def run_script():
    # Get the absolute path of the target script
    script_path = os.path.abspath("src/run_loop/run_closed_loop.py")

    # Check if the script exists
    if not os.path.isfile(script_path):
        print(f"Error: Script not found at {script_path}")
        sys.exit(1)

    # Run the target script using python with unbuffered output (-u)
    try:
        result = subprocess.run(
            [sys.executable, "-u", script_path],
            stdout=sys.stdout,
            stderr=sys.stderr,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"Script {script_path} failed with return code {e.returncode}")
        sys.exit(e.returncode)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

# Ensures the script runs only when executed directly
if __name__ == "__main__":
    run_script()
