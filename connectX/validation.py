import torch
from utilities import process_state
from kaggle_environments import evaluate

def validate(Q_func, num_episodes=100, op_agent='random', device='cpu'):
    Q_func.eval()

    def mean_reward(rewards):
        return sum(r[0] for r in rewards) / float(len(rewards))


    def my_agent(obs, env_config):
        assert env_config.rows == 6
        assert env_config.columns == 7
        assert env_config.inarow == 4

        moves = Q_func(process_state(obs, device)[None,:])[0]
        mask_board = torch.tensor(obs.board[:env_config.columns], dtype=bool, device=device)

        action = torch.masked_fill(moves, mask_board, -float('inf')).argmax().item()
        print('action:', action)
        return action
    
    val_reward = mean_reward(evaluate("connectx", [my_agent, op_agent], num_episodes=num_episodes))
    Q_func.train()

    return val_reward
    

