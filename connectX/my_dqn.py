import torch


class My_DQN(torch.nn.Module):
    def __init__(self, env_config, my_config, device='cpu'):
        super().__init__()
        self.env_config = env_config

        self.model = torch.nn.Sequential(
            torch.nn.Linear(env_config.rows * env_config.columns, my_config.h, device=device),
            torch.nn.ReLU(),
            torch.nn.Linear(my_config.h, my_config.h, device=device),
            torch.nn.ReLU(),
            torch.nn.Linear(my_config.h, my_config.h, device=device),
            torch.nn.ReLU(),
            torch.nn.Linear(my_config.h, env_config.columns, device=device)
        )

    def forward(self, X):
        assert X.shape[-1] == self.env_config.rows * self.env_config.columns
        return self.model(X.to(dtype=torch.float32))
