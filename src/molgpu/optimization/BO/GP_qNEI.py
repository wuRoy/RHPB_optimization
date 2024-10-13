from botorch.models import SingleTaskGP
from gpytorch.mlls.sum_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from molgpu.utils.utils import selected_enumeration
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.utils.transforms import standardize
from molgpu.optimization.BO.utils import (
    get_data,
    model_evaluation,
    set_seeds
    )
from sklearn.model_selection import train_test_split
import torch
import pandas as pd
import numpy as np


class GP_qNEI:
    def __init__(
            self,
            project_dir,
            num_candidates,
            iteration,
            ):
        self.project_dir = project_dir
        self.num_candidates = num_candidates
        self.iteration = iteration
        x, y = get_data(self.project_dir)
        self.x = x.to('cpu')
        self.y = standardize(y).unsqueeze(-1).to('cpu')

    def initialize_model(
            self,
            train_x,
            train_y,
            state_dict=None
            ):
        # build the model
        model = SingleTaskGP(train_x, train_y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        if state_dict is not None:
            model.load_state_dict(state_dict)
        return mll, model

    def optimize_discrete_EI(
        self,
        acq_func
    ) -> np.ndarray:
        '''
            Optimize the acquisition function and return the new candidates
        '''
        d = self.x.shape[1]
        enumerate_choices = selected_enumeration(4, d)
        # transform the enumerate_choices to tensor
        enumerate_choices = torch.from_numpy(enumerate_choices).to(self.x).double()
        batches = enumerate_choices.reshape(
            enumerate_choices.shape[0],
            1,
            enumerate_choices.shape[-1]
        )
        acqu_values_batch = acq_func(batches)
        acqu_values_batch = acqu_values_batch.flatten()
        top_idx = torch.topk(
            acqu_values_batch,
            k=self.num_candidates,
            largest=True
        )[1]
        candidates = enumerate_choices[top_idx]
        candidates = candidates.cpu().numpy()

        return candidates

    def train_and_propose(self):
        '''
            Train the GP model and propose new candidates
        '''
        set_seeds(42)
        X_train, X_test, y_train, y_test = train_test_split(
            self.x, self.y, test_size=0.1, random_state=42
        )

        mll, model = self.initialize_model(self.x, self.y)
        print('Training the GP model')
        fit_gpytorch_mll(mll)
        model_eval = model_evaluation(model, X_test, y_test)
        # save r_2 and mse to a csv file
        model_eval_df = pd.DataFrame([model_eval])
        model_eval_df.to_csv(self.project_dir / 'data' / f'round_{self.iteration}' / 'model_eval.csv', index=False)

        EI = LogExpectedImprovement(
                    model=model,
                    best_f=self.y.max()
                )
        print('Proposing new candidates...')
        new_x = self.optimize_discrete_EI(EI)
        # transform the new_x to numpy array
        new_x = new_x.round(2)
        return new_x
