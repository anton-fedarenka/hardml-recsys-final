import numpy as np
from typing import List
from dataclasses import dataclass

@dataclass
class ThompsonSamplingBandit():
    """
    The class implements the functionality of training and applying 
    the multi-armed bandit algorithm in the Thompson sampling version.
    This class is implemented as a dataclass to be able to store 
    intermediate bandit states in a database like Redis.
    """
    alpha: List[float] # = field(default_factory = lambda: np.zeros(n_arms) + init_alpha)
    beta:  List[float] #= field(default_factory = lambda: np.zeros(n_arms) + init_beta)
    alpha_weight: float = 1.
    beta_weight: float = 1.
        
    def sample(self, rng_seed: int = None) -> int:
        rng = np.random.default_rng(rng_seed)
        return np.argmax(
            rng.beta(
                self.alpha_weight * np.array(self.alpha), 
                self.beta_weight * np.array(self.beta)
            )
        )

    def get_top_indices(self, top_k: int = 10, n_bunch: int = 50, rng_seed: int = None) -> List[int]:
        '''
        Produces several (n bunch) different samples from the beta distribution, each of size top_k.
        '''
        rng = np.random.default_rng(rng_seed)
        beta_distr_array = rng.beta(
            self.alpha_weight * np.array(self.alpha), 
            self.beta_weight * np.array(self.beta),
            size=(n_bunch, len(self.alpha))
        )
        top_n_inds = np.argpartition(beta_distr_array, -top_k)[:, -top_k:]
        top_n_vals = np.take_along_axis(beta_distr_array, top_n_inds, axis=1)
        return np.take_along_axis(top_n_inds, np.argsort(top_n_vals)[:,::-1], axis=1).tolist() 
    
    def retrieve_reward(self, arm_ind: int, action: str, n: int) -> None:
        '''
        Updates bandit state according to the data received.
        '''
        if action == 'like':
            self.alpha[arm_ind] += n
        else:
            self.beta[arm_ind] += n
        return


def create_bandit_instance(n_arms: int, alpha_weight: int = 1, beta_weight: int = 1):
    '''
    Creates a new bandit instance.
    '''
    return ThompsonSamplingBandit(
        alpha = np.ones(n_arms).tolist(),
        beta =  np.ones(n_arms).tolist(),
        alpha_weight=alpha_weight, 
        beta_weight=beta_weight
        )
