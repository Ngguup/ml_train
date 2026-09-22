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
    

def next_op_agent(train_config):
    op_agents = train_config.op_agents
    i = 0
    while True:
        yield op_agents[i]
        i = (i + 1) % len(op_agents)


# make submission file
import inspect
from my_dqn import My_DQN
from types import SimpleNamespace


def my_agent(obs, env_config):
    assert env_config.rows == 6
    assert env_config.columns == 7
    assert env_config.inarow == 4

    my_config = SimpleNamespace(
        h=128
    )

    Q_func = My_DQN(env_config, my_config)
    Q_func.load_state_dict(state_dict)

    moves = Q_func(process_state(obs)[None,:])[0]
    mask_board = torch.tensor(obs.board[:env_config.columns], dtype=bool)

    action = torch.masked_fill(moves, mask_board, -float('inf')).argmax().item()
    print('action:', action)
    return action

def make_submission(Q_func):
    state_dict = Q_func.state_dict()

    with open('submission.py', 'w') as f:
        f.write('import torch\n')
        f.write('from types import SimpleNamespace\n')
        f.write('from collections import OrderedDict\n')
        f.write('\n\n')

        f.write('state_dict = OrderedDict({\n')
        for name, tensor in state_dict.items():
            f.write(f'\t{name!r}: torch.tensor(')
            f.write(repr(tensor.cpu().tolist()))
            f.write(f', dtype=torch.{str(tensor.dtype).split('.')[-1]}),\n')
        f.write('})\n\n')

        f.write(f'{inspect.getsource(My_DQN)}\n\n')
        f.write(f'{inspect.getsource(process_state)}\n\n')
        f.write(f'{inspect.getsource(my_agent)}')
