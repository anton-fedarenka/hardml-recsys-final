import numpy as np
from typing import List
from dataclasses import dataclass

@dataclass
class ThompsonSamplingBandit():
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

    def get_top_indices(self, k: int = 10, rng_seed: int = None) -> List[int]:
        rng = np.random.default_rng(rng_seed)
        beta_distr_array = rng.beta(
            self.alpha_weight * np.array(self.alpha), 
            self.beta_weight * np.array(self.beta)
        )
        top_n_inds = np.argpartition(beta_distr_array, -k)[-k:]
        top_n_vals = beta_distr_array[top_n_inds]
        return top_n_inds[np.argsort(top_n_vals)[::-1]].tolist() 
    
    def retrieve_reward(self, arm_ind: int, action: str, n: int) -> None:
        if action == 'like':
            self.alpha[arm_ind] += n
        else:
            self.beta[arm_ind] += n
        return


def create_bandit_instance(n_arms: int, alpha_weight: int = 1, beta_weight: int = 1):
    return ThompsonSamplingBandit(
        alpha = np.ones(n_arms).tolist(),
        beta =  np.ones(n_arms).tolist(),
        alpha_weight=alpha_weight, 
        beta_weight=beta_weight
        )
