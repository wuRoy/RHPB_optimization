import numpy as np
from pathlib import Path
import torch
import sklearn.metrics
from sklearn.model_selection import train_test_split
import pandas as pd

# could be removed


def model_evaluation(model, X_test, y_test):
    # evaluate the model
    model.eval()
    y_hat = model(X_test).mean.cpu().detach().numpy()
    r_2 = sklearn.metrics.r2_score(y_hat, y_test.cpu().numpy())
    mse = sklearn.metrics.mean_squared_error(y_hat, y_test.cpu().numpy())
    print("Train R2 score:", r_2)
    print("Train MSE score:", mse)
    return {'r_2': r_2, 'mse': mse}


def get_data(project_dir: Path):
    '''
        Get all the data from the csv files in the project directory
    '''
    # Get all the csv files in the data directory
    csv_files = sorted(Path(project_dir / 'data').glob('round_*/combined_results.csv'))
    # Initialize an empty list to hold the tensors
    tensors = []
    # Loop through all csv files and read them into tensors
    for csv_file in csv_files:
        arr = np.genfromtxt(csv_file, delimiter=',')
        arr = arr[8:, :]
        tensor = torch.from_numpy(arr)
        tensors.append(tensor)

    # if there is previous data, read it
    previous_data = project_dir / 'data' / 'previous_data.csv'
    if previous_data.exists():
        # read the data from the csv file
        df = pd.read_csv(previous_data)
        # remove the rows whose compositions are all zeros (controls)
        df = df[(df.iloc[:, :-2] != 0).any(axis=1)]
        df = df.to_numpy()
        df = df[:, :-1]
        tensor = torch.from_numpy(df)
        tensors.append(tensor)

    # Concatenate all tensors into one
    combined_tensor = torch.cat(tensors)
    x = combined_tensor[:, :-1]
    y = combined_tensor[:, -1]
    # remove repeated rows
    x, idx = np.unique(x, axis=0, return_index=True)
    y = y[idx]
    x = torch.tensor(x)

    return x, y


def set_seeds(seed: int = 42) -> None:
    '''
        Set the seeds for reproducibility
    '''
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



