import torch


def compute_loss(minibatch, Q_func, Q_func_hat, train_config, device='cpu'):
    states, actions, rewards, next_states, dones = minibatch
    with torch.no_grad():
        target_rewards = rewards + Q_func_hat(next_states).max(dim=-1).values * ~dones
    pred_rewards = Q_func(states)[torch.arange(train_config.batch_size, device=device),actions]

    loss = (1 / train_config.batch_size) * torch.square(target_rewards - pred_rewards).sum()
    return loss