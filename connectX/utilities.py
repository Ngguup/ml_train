import torch


def process_state(obs, device='cpu'):
    board = torch.tensor(obs.board, device=device)

    opponent_mark = 2 if obs.mark == 1 else 1
    board[board==opponent_mark] = -1
    board[board==obs.mark] == 1

    return board

class Memory:
    def __init__(self, env_config, train_config, device='cpu'):
        self.train_config = train_config

        self.state = torch.zeros(
            train_config.total_steps, 
            env_config.rows * env_config.columns,
            device=device
        )
        self.action = torch.zeros(train_config.total_steps, device=device, dtype=torch.int64)
        self.reward = torch.zeros(train_config.total_steps, device=device)
        self.next_state = torch.zeros(
            train_config.total_steps, 
            env_config.rows * env_config.columns,
            device=device
        )
        self.done = torch.zeros(train_config.total_steps, device=device, dtype=bool)

        self._at = 0

    def __len__(self):
        return self._at

    def write(self, state, action, reward, next_state, done):
        at = self._at
        self.state[at] = state
        self.action[at] = action
        self.reward[at] = reward
        self.next_state[at] = next_state
        self.done[at] = done

        self._at += 1
        
    def sample_minibatch(self):
        idx = torch.randperm(self._at)[:self.train_config.batch_size]

        return (
            self.state[idx],
            self.action[idx],
            self.reward[idx],
            self.next_state[idx],
            self.done[idx]
        )