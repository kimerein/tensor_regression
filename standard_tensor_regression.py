import numpy as np
import copy
import matplotlib.pyplot as plt

import torch
import torch.cuda
from torch.autograd import Variable
from torch.optim import LBFGS

import tensorly as tl

####################################
######## Useful functions ##########
####################################

def set_device(use_GPU=True, verbose=True):
    """
    Set torch.cuda device to use.
    RH2021

    Args:
        use_GPU (int):
            If 1, use GPU.
            If 0, use CPU.
    """
    if use_GPU:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device != "cuda":
            print("no GPU available. Using CPU.") if verbose else None
        else:
            print("GPU is enabled.") if verbose else None
    else:
        device = "cpu"
        print("using CPU") if verbose else None

    return device
    
def squeeze_integers(arr):
    """
    Make integers in an array consecutive numbers
     starting from 0. ie. [7,2,7,4,1] -> [3,2,3,1,0].
    Useful for removing unused class IDs from y_true
     and outputting something appropriate for softmax.
    RH2021

    Args:
        arr (np.ndarray):
            array of integers.
    
    Returns:
        arr_squeezed (np.ndarray):
            array of integers with consecutive numbers
    """
    uniques = np.unique(arr)
    arr_squeezed = copy.deepcopy(arr)
    for val in np.arange(0, np.max(arr)+1):
        if np.isin(val, uniques):
            continue
        else:
            arr_squeezed[arr_squeezed>val] = arr_squeezed[arr_squeezed>val]-1
    return arr_squeezed


####################################
######## Helper functions ##########
####################################

def make_BcpInit(B_dims, rank, non_negative, scale=1, device='cpu'):
    """
    Make initial Beta Kruskal tensor.
    RH2021
    
    Args:
        B_dims (list of ints):
            List of dimensions of each component.
        rank (int):
            Rank of each component.
        non_negative (list of booleans):
            List of booleans indicating whether each component
             is non-negative.
        scale (float):
            Scale of uniform distribution used to initialize
             each component.
        device (str):
            Device to use.
    
    Returns:
        B_cp (list of torch.Tensor):
            Beta Kruskal tensor.
    """
    Bcp_init = [(torch.rand((B_dims[ii], rank))*scale - non_negative[ii]*(scale/2)).to(device) for ii in range(len(B_dims))]
    for ii in range(len(B_dims)):
        Bcp_init[ii].requires_grad = True
    return Bcp_init

def non_neg_fn(B_cp, non_negative, softplus_kwargs=None):
    """
    Apply softplus to specified dimensions of Bcp.
    Generator function that yields a list of tensors.
    RH2021

    Args:
        B_cp (list of torch.Tensor):
            Beta Kruskal tensor (before softplus).
            List of tensors of shape
             (n_features, rank).
        non_negative (list of booleans):
            List of booleans indicating whether each component
             is non-negative.
        softplus_kwargs (dict):
            Keyword arguments for torch.nn.functional.softplus.
    
    Yields:
        B_cp_nonNegative (list of torch.Tensor):
            Beta Kruskal tensor with softplus applied to
             specified dimensions.
    """
    if softplus_kwargs is None:
        softplus_kwargs = {
            'beta': 50,
            'threshold': 1,
        }
    for ii in range(len(B_cp)):
        if non_negative[ii]:
            yield torch.nn.functional.softplus(B_cp[ii], **softplus_kwargs)
        else:
            yield B_cp[ii]

def lin_model(X, Bcp, weights, non_negative, softplus_kwargs=None):
    """
    Compute the regression model.
    y_hat = inner(X, outer(softplus(Bcp))
    where:
        X.shape[1:] == Bcp.shape[:-1]
        X.shape[0] == len(y_hat)
        Bcp.shape[-1] == len(unique(y_true))
        softplus is performed only on specified dimensions of Bcp.
        inner prod is performed on dims [1:] of X and
         dims [:-1] of Bcp.
    JZ2021

    Args:
        X (torch.Tensor):
            N-D array of data.
        Bcp (list of torch.Tensor):
            Beta Kruskal tensor (before softplus).
            List of tensors of shape 
             (n_features, rank).
        weights (list of floats):
            List of weights for each component.
            len(weights) == rank == Bcp[0].shape[1]
        non_negative (list of booleans):
            List of booleans indicating whether each component
             is non-negative.
        softplus_kwargs (dict):
            Keyword arguments for torch.nn.functional.softplus.
    
    Returns:
        y_hat (torch.Tensor):
            N-D array of predictions.
    """
    
    return tl.tenalg.inner(X,
                           tl.cp_tensor.cp_to_tensor((weights, list(non_neg_fn(
                                                                Bcp,
                                                                non_negative,
                                                                softplus_kwargs))
                                                     )),
                           n_modes=len(Bcp)-1
                        )
    
        
def L2_penalty(B_cp):
    """
    Compute the L2 penalty.
    RH2021

    Args:
        B_cp (list of torch.Tensor):
            Beta Kruskal tensor (before softplus)
    
    Returns:
        L2_penalty (torch.Tensor):
            L2 penalty.
    """
    ii=0
    for comp in B_cp:
        ii+= torch.sqrt(torch.sum(comp**2))
    return ii


####################################
########### Main class #############
####################################

class CP_linear_regression():
    def __init__(self, X, y, rank=5, non_negative=False, weights=None, Bcp_init=None, Bcp_init_scale=1, device='cpu', softplus_kwargs=None):
        """
        Multinomial logistic CP tensor regression class.
        Bias is not considered in this model because there
         are multiple ways to handle bias with tensor
         regression. User must add a bias entry to X
         manually if they wish. Mean subtracting X along
         one dimension is recommended.
        RH 2021

        Args:
            X (np.ndarray or torch.Tensor):
                Input data. First dimension must match len(y).
            y (np.ndarray or torch.Tensor):
                Class labels. Length must match first dimension of X.
            rank (int):
                Rank of the CP (Kruskal) tensor used to compute the
                 beta tensor that is multiplied by X.
            non_negative (False or True or list):
                If False, the CP tensor is allowed to be negative.
                If True, the CP tensor is forced to be non-negative.
                If a list, then the list should be length X.ndim,
                 and each element should be a boolean indicating 
                 whether that dimension of the beta tensor should be
                 forced to be non-negative.
                 Entries correspond to the dimension index+1 of X, 
                 and last dim is the classification (one hot) dim of
                 the beta tensor.
            weights (np.ndarray or torch.Tensor):
                Weights for each component. Should be a
                 vector of length(rank).
            Bcp_init (list):
                Optional. List of initial Bcp matrices.
                If None, then the initial Bcp matrices are set as 
                 random number from a uniform distribution between
                 either 0 to 1 if that dimension is non_negative or
                 -1 to 1 if dimension is not non_negative.
            Bcp_init_scale (float):
                Optional. Scale of uniform distribution used to
                 initialize each component of Bcp.
            device (str):
                Device to run the model on.
        """        

        self.X = torch.tensor(X, dtype=torch.float32).to(device)
        self.y = torch.tensor(y, dtype=torch.float32).to(device)
        
        if weights is None:
            self.weights = torch.ones((rank), device=device)
        else:
            self.weights = torch.tensor(weights)

        if softplus_kwargs is None:
            self.softplus_kwargs = {'beta': 50, 'threshold': 1}
        else:
            self.softplus_kwargs = softplus_kwargs

        self.rank = rank
        self.device = device

        if non_negative == True:
            self.non_negative = [True]*(self.X.ndim)
        elif non_negative == False:
            self.non_negative = [False]*(self.X.ndim)
        else:
            self.non_negative = non_negative        

        
        self.n_classes = len(torch.unique(self.y))
        B_dims = np.concatenate((np.array(self.X.shape[1:]), [1]))
        if Bcp_init is None:
            self.Bcp = make_BcpInit(B_dims, self.rank, self.non_negative, scale=Bcp_init_scale, device=self.device)
        else:
            self.Bcp = Bcp_init

        self.loss_running = []

    def fit(self,
            lambda_L2=0.01, 
            max_iter=1000, 
            tol=1e-5, 
            patience=10,
            verbose=False,
            running_loss_logging_interval=10, 
            LBFGS_kwargs=None):
        """
        Fit a beta tensor (self.Bcp) to the data using the
         LBFGS optimizer.
        Note that self.Bcp is not the final Kruskal tensor, 
         non_neg_fn(self.Bcp, non_negative) 
         is the final Kruskal tensor.
        Use self.return_Bcp_final() to get the final tensor.
        Note that logging the loss (self.loss_running)
         requires running the model extra times. If data is
         large, then set running_loss_logging_interval to a
         large number.
        RH 2021

        Args:
            lambda_L2 (float):
                L2 regularization parameter.
            max_iter (int):
                Maximum number of iterations.
            tol (float):
                Tolerance for the stopping criterion.
            patience (int):
                Number of iterations with no improvement to wait
                 before early stopping.
            verbose (0, 1, or 2):
                If 0, then no output.
                If 1, then only output whether the model has
                 converged or not.
                If 2, then output the loss at each iteration.
            running_loss_logging_interval (int):
                Number of iterations between logging of running loss.
            LBFGS_kwargs (dict):
                Keyword arguments for LBFGS optimizer.
        
        Returns:
            convergence_reached (bool):
                True if convergence was reached.
        """

        if LBFGS_kwargs is None:
            {
                'lr' : 1, 
                'max_iter' : 20, 
                'max_eval' : 20, 
                'tolerance_grad' : 1e-07, 
                'tolerance_change' : 1e-09, 
                'history_size' : 100, 
                'line_search_fn' : "strong_wolfe"
            }

        tl.set_backend('pytorch')

        optimizer = torch.optim.LBFGS(self.Bcp, **LBFGS_kwargs)

        def closure():
            optimizer.zero_grad()
            y_hat = lin_model(self.X, self.Bcp, self.weights, self.non_negative, softplus_kwargs=self.softplus_kwargs)
            loss = loss_fn(y_hat.reshape(-1), self.y.reshape(-1)) + lambda_L2 * L2_penalty(self.Bcp)
            loss.backward()
            return loss
            
        loss_fn = torch.nn.MSELoss()

        convergence_reached = False
        for ii in range(max_iter):
            if ii%running_loss_logging_interval == 0:
                y_hat = lin_model(self.X, self.Bcp, self.weights, self.non_negative, softplus_kwargs=self.softplus_kwargs)
                self.loss_running.append(loss_fn(y_hat.reshape(-1), self.y.reshape(-1)).item())
                if verbose==2:
                    print(f'Iteration: {ii}, Loss: {self.loss_running[-1]}')

            if ii > patience:
                if np.sum(np.abs(np.diff(self.loss_running[ii-patience:]))) < tol:
                    convergence_reached = True
                    break

            optimizer.step(closure)
        if (verbose==True) or (verbose>=1):
            if convergence_reached:
                print('Convergence reached')
            else:
                print('Reached maximum number of iterations without convergence')
        return convergence_reached

    def fit_Adam(self,
            lambda_L2=0.01, 
            max_iter=1000, 
            tol=1e-5, 
            patience=10,
            verbose=False,
            Adam_kwargs=None):
        """
        Fit a beta tensor (self.Bcp) to the data using the
         Adam optimizer.
        Note that self.Bcp is not the final Kruskal tensor, 
         non_neg_fn(self.Bcp, non_negative) 
         is the final Kruskal tensor.
        Use self.return_Bcp_final() to get the final tensor.
        Note that logging the loss (self.loss_running)
         requires running the model extra times. If data is
         large, then set running_loss_logging_interval to a
         large number.
        RH 2021

        Args:
            lambda_L2 (float):
                L2 regularization parameter.
            max_iter (int):
                Maximum number of iterations.
            tol (float):
                Tolerance for the stopping criterion.
            patience (int):
                Number of iterations with no improvement to wait
                 before early stopping.
            verbose (0, 1, or 2):
                If 0, then no output.
                If 1, then only output whether the model has
                 converged or not.
            Adam_kwargs (dict):
                Keyword arguments for Adam optimizer.
        
        Returns:
            convergence_reached (bool):
                True if convergence was reached.
        """

        if Adam_kwargs is None:
            {
                'lr' : 1, 
                'betas' : (0.9, 0.999), 
                'eps' : 1e-08, 
                'weight_decay' : 0, 
                'amsgrad' : False
            }

        tl.set_backend('pytorch')

        optimizer = torch.optim.Adam(self.Bcp, **Adam_kwargs)
        loss_fn = torch.nn.MSELoss()

        convergence_reached = False
        for ii in range(max_iter):
            optimizer.zero_grad()
            y_hat = lin_model(self.X, self.Bcp, self.weights, self.non_negative, softplus_kwargs=self.softplus_kwargs)
            loss = loss_fn(y_hat, self.y) + lambda_L2 * L2_penalty(self.Bcp)
            loss.backward()
            optimizer.step()
            self.loss_running.append(loss.item())
            if verbose==2:
                print(f'Iteration: {ii}, Loss: {self.loss_running[-1]}')
            if ii > patience:
                if np.sum(np.abs(np.diff(self.loss_running[ii-patience:]))) < tol:
                    convergence_reached = True
                    break
        if (verbose==True) or (verbose>=1):
            if convergence_reached:
                print('Convergence reached')
            else:
                print('Reached maximum number of iterations without convergence')
        return convergence_reached


    def predict(self, X=None, y_true=None, Bcp=None, device=None, plot_pref=False):
        """
        Predict class labels for X given a Bcp (beta Kruskal tensor).
        Uses 'model' function in this module.
        RH 2021

        Args:
            X (np.ndarray or torch.Tensor):
                Input data. First dimension must match len(y).
                If None, then use the data that was passed to the
                 constructor.
            Bcp (list of np.ndarray or torch.Tensor):
                List of Bcp matrices.
                If None, then use the Bcp that was passed to the
                 constructor.
            y_true (np.ndarray or torch.Tensor):
                True class labels. Only necessary if plot_pref is
                 True and inputting data from a different X than
                 what the model was trained on.
            device (str):
                Device to run the model on.
            plot_pref (bool):
                If True, then make plots
        
        Returns:
            logits (np.ndarray):
                Raw output of the model.
            preds (np.ndarray):
                Predicted class labels.
            cm (np.ndarray):
                Confusion matrix.
            acc (float):
                Accuracy.
        """
        if device is None:
            device = self.device
        
        if X is None:
            X = self.X
        elif isinstance(X, torch.Tensor) == False:
            X = torch.tensor(X, dtype=torch.float32, requires_grad=False).to(device)
        elif X.device != device:
            X = X.to(device)

        if y_true is None:
            y_true = self.y.detach().cpu().numpy()

                        
        if Bcp is None:
            Bcp = self.Bcp
        elif isinstance(Bcp[0], torch.Tensor) == False:
            for ii in range(len(Bcp)):
                Bcp[ii] = torch.tensor(Bcp[ii], dtype=torch.float32, requires_grad=False).to(device)
        elif Bcp[0].device != device:
            for ii in range(len(Bcp)):
                Bcp[ii] = Bcp[ii].to(device)

        y_hat = lin_model(X, Bcp, self.weights, self.non_negative, softplus_kwargs=self.softplus_kwargs).detach().cpu().numpy()
            
        return y_hat

    
    def return_Bcp_final(self):
        """
        Return the final Kruskal tensor as a numpy array.
        Simply passes self.Bcp to non_neg_fn.
        RH 2021

        Returns:
            Bcp_nonNeg (np.ndarray):
                Final Kruskal tensor. Ready to multiply by the
                 data to predict class labels (y_hat).
        """
        Bcp = list(non_neg_fn(self.Bcp, self.non_negative, softplus_kwargs=self.softplus_kwargs))
        Bcp_nonNeg = [Bcp[ii].detach().cpu().numpy() for ii in range(len(Bcp))]
        return Bcp_nonNeg
    
    def get_params(self):
        """
        Get the parameters of the model.
        RH 2021
        """
        return {'X': self.X,
                'y': self.y,
                'weights': self.weights,
                'Bcp': self.Bcp,
                'non_negative': self.non_negative,
                'softplus_kwargs': self.softplus_kwargs,
                'rank': self.rank,
                'device': self.device}

    def set_params(self, params):
        """
        Set the parameters of the model.
        RH 2021

        Args:
            params (dict):
                Dictionary of parameters.
        """
        self.X = params['X']
        self.y = params['y']
        self.weights = params['weights']
        self.Bcp = params['Bcp']
        self.non_negative = params['non_negative']
        self.softplus_kwargs = params['softplus_kwargs']
        self.rank = params['rank']
        self.device = params['device']

    def display_params(self):
        """
        Display the parameters of the model.
        RH 2021
        """
        print('X:', self.X.shape)
        print('y:', self.y.shape)
        print('weights:', self.weights)
        print('Bcp:', self.Bcp)
        print('non_negative:', self.non_negative)
        print('softplus_kwargs:', self.softplus_kwargs)
        print('rank:', self.rank)
        print('device:', self.device)
        
    def plot_outputs(self):
        """
        Plot the outputs of the model.
        RH 2021
        """
        plt.figure()
        plt.plot(self.loss_running)
        plt.xlabel('logged iteration')
        plt.ylabel('loss')
        plt.title('loss')

        Bcp_final = self.return_Bcp_final()
        fig, axs = plt.subplots(len(Bcp_final))
        for ii, val in enumerate(Bcp_final):
            axs[ii].set_title(f'factor {ii}')
            axs[ii].plot(val)
        fig.suptitle('components')


if __name__ == '__main__':
    import torch
    import numpy as np
    import tensorly as tl
    import scipy.signal
    import matplotlib.pyplot as plt

    import sys
    sys.path.append(r'..')

    import tensorly as tl

    torch.manual_seed(321)
    np.random.seed(321)

    X_dims_fake = [2000, 500, 500]
    nClasses_fake = 5

    # y_true  = np.random.randint(0, nClasses_fake, X_dims_fake[0])
    # y_true_oneHot = mtr.idx_to_oneHot(y_true, nClasses_fake)

    Xcp_underlying_fake = [
                        torch.rand(X_dims_fake[0], 4)-0.5,
                        torch.vstack([torch.sin(torch.linspace(0, 140, X_dims_fake[1])),
                                        torch.cos(torch.linspace(2,19,X_dims_fake[1])),
                                        torch.linspace(0,1,X_dims_fake[1]),
                                        torch.cos(torch.linspace(0,17,X_dims_fake[1])) >0]).T ,
                        torch.tensor(scipy.signal.savgol_filter(np.random.rand(X_dims_fake[2], 4), 15, 3, axis=0))-0.5,
                        ]
    # Bcp_underlying_fake = Xcp_underlying_fake[1:] + [torch.rand(nClasses_fake, 4) -0.5]
    Bcp_underlying_fake = Xcp_underlying_fake[1:] + [torch.ones(1, 4) -0.5]

    tl.set_backend('pytorch')
    X_fake = tl.cp_tensor.cp_to_tensor((np.ones(4), Xcp_underlying_fake))

    y_hat = tl.tenalg.inner(X_fake,
                        tl.cp_tensor.cp_to_tensor((np.ones(4), Bcp_underlying_fake )),
                        n_modes=len(Bcp_underlying_fake)-1)

    X = X_fake.numpy()
    X = (X - np.mean(X, axis=1)[:,None,:])
    y = y_hat.numpy()
    DEVICE = set_device(use_GPU=True)

    # h_vals = np.logspace(-50, 2, num=30, endpoint=True, base=10.0)
    # h_vals = np.int64(np.linspace(1, 300, num=30, endpoint=True))
    h_vals = np.arange(1)

    loss_all = []
    for ii, val in enumerate(h_vals):
        print(f'hyperparameter val: {val}')
        cpmlr = CP_linear_regression(X, y, 
                                            rank=1,
                                            non_negative=[False, False, False],
                                            weights=None,
                                            Bcp_init=None,
                                            Bcp_init_scale=1,
                                            device=DEVICE,
                                            softplus_kwargs={
                                                'beta': 50,
                                                'threshold':1}
                                            )

        # tic = time.time()
        cpmlr.fit(lambda_L2=0.003, 
                    max_iter=200, 
                    tol=1e-50, 
                    patience=10,
                    verbose=1,
                    running_loss_logging_interval=1,
                    LBFGS_kwargs={
                        'lr' : 1, 
                        'max_iter' : 20, 
                        'max_eval' : None, 
                        'tolerance_grad' : 1e-07, 
                        'tolerance_change' : 1e-09, 
                        'history_size' : 100, 
                        'line_search_fn' : "strong_wolfe"
                    }
                )

        print(f'loss: {cpmlr.loss_running[-1]}')
        
        loss_all.append(cpmlr.loss_running[-1])

    pass