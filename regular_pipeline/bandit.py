import numpy as np
from typing import List

class BaseBandit:
    def sample(self) -> int:
        raise NotImplementedError()

    def retrieve_reward(self, ind: int, reward: float):
        raise NotImplementedError()
    
class ThompsonSamplingBandit(BaseBandit):
    def __init__(
        self,
        n_arms: int,
        init_alpha: float = 1.,
        init_beta: float = 1.,
        alpha_weight: float = 1.,
        beta_weight: float = 1.,
    ):
        super().__init__()
        
        self.cur_iteration = 0
        
        self.n_arms = n_arms
        self.alpha = np.zeros(n_arms) + init_alpha
        self.beta = np.zeros(n_arms) + init_beta
        
        self.alpha_weight = alpha_weight
        self.beta_weight = beta_weight
        
    def sample(self) -> int:
        return np.argmax(np.random.beta(self.alpha_weight * self.alpha, self.beta_weight * self.beta))

    def get_top_indices(self, k: int = 10) -> List[int]:
        beta_distr_array = np.random.beta(self.alpha_weight * self.alpha, self.beta_weight * self.beta)
        top_n_inds = np.argpartition(beta_distr_array, -k)[-k:]
        top_n_vals = beta_distr_array[top_n_inds]
        return top_n_inds[np.argsort(top_n_vals)[::-1]].tolist() 
    
    def retrieve_reward(self, arm_ind: int, action: str, n: int) -> None:
        if action == 'like':
            self.alpha[arm_ind] += n
        else:
            self.beta[arm_ind] += n
        return