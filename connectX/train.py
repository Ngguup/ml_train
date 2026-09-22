import torch
import numpy as np
from utilities import Memory, process_state
from kaggle_environments import make
from reward import compute_reward_soft
from validation import validate

def train(Q_func, optimizer, scheduler, my_config, env_config, train_config, op_agent='random', device='cpu'):
    env = make('connectx')

    logs = {'loss': [], 'val_reward': [], 'lr': []}
    memory = Memory(env_config, train_config, device)

    Q_func_hat = type(Q_func)(env_config, my_config, device)
    Q_func_hat.load_state_dict(Q_func.state_dict())

    Q_func.train()
    Q_func_hat.eval()

    trainer = env.train([None, op_agent])
    done = False
    obs = trainer.reset()
    state = process_state(obs, device)

    for step in range(train_config.total_steps):
        if torch.rand(1) <= train_config.epsilon:
            action = int(np.random.choice(
                [c for c in range(env_config.columns) if state[c] == 0]
            ))
        else:
            action = Q_func(state[None,:]).argmax(dim=-1).item()

        obs, reward, done, _ = trainer.step(action)
        next_state = process_state(obs, device)
        # if reward is None: reward = train_config.reward_for_None
        reward = compute_reward_soft(reward, train_config)
        # reward = compute_reward_strict(reward, state, next_state, env_config, train_config, device)

        memory.write(state, action, reward, next_state, done)

        if len(memory) < train_config.batch_size * 2 and step % 100 == 0:
            print(f'| step: {step} |')

        if len(memory) >= train_config.batch_size * 2: # ???
            states, actions, rewards, next_states, dones = memory.sample_minibatch()
            
            with torch.no_grad():
                target_rewards = rewards + Q_func_hat(next_states).max(dim=-1).values * ~dones
            pred_rewards = Q_func(states)[torch.arange(train_config.batch_size, device=device),actions]

            loss = (1 / train_config.batch_size) * torch.square(target_rewards - pred_rewards).sum()
            logs['loss'].append(loss.item())
            logs['lr'].append(optimizer.param_groups[0]['lr'])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if step % 100 == 0:
                print(f'| step: {step} | loss: {loss.item()} |')

        if done:
            obs = trainer.reset()
            state = process_state(obs, device)
        else:
            state = next_state

        if step % train_config.validate_every == 0:
            val_reward = validate(
                Q_func, 
                num_episodes=train_config.num_episodes,
                op_agent=op_agent, 
                device=device
            )
            logs['val_reward'].append(val_reward)

            print(f'| step: {step} | val_reward: {val_reward} |')
    
    return logs