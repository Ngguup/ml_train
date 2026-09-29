import torch
import numpy as np
from utilities import Memory, process_state, next_op_agent
from kaggle_environments import make
from reward import compute_reward_soft
from validation import validate
from loss import compute_loss


def train(Q_func, optimizer, scheduler, my_config, env_config, train_config, device='cpu'):
    env = make('connectx')

    logs = {'loss': [], 'val_reward': [], 'val_step': [], 'lr': []}
    memory = Memory(env_config, train_config, device)

    Q_func_hat = type(Q_func)(env_config, my_config, device)
    Q_func_hat.load_state_dict(Q_func.state_dict())

    Q_func.train()
    Q_func_hat.eval()

    op_agents = next_op_agent(train_config)
    trainer = env.train([None, next(op_agents)])
    obs = trainer.reset()
    done = False
    state = process_state(obs, device)

    for step in range(1, train_config.total_steps + 1):
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

        if len(memory) < train_config.batch_size * train_config.train_after_batch_n and step % 100 == 0:
            print(f'| step: {step} |')

        if len(memory) >= train_config.batch_size * train_config.train_after_batch_n: # ???
            minibatch = memory.sample_minibatch()
            loss = compute_loss(minibatch, Q_func, Q_func_hat, train_config, device=device)

            logs['loss'].append(loss.item())
            logs['lr'].append(optimizer.param_groups[0]['lr'])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if step % 100 == 0:
                print(f'| step: {step} | loss: {loss.item():.4f} |')

            # move target
            if step % train_config.move_target_every == 0:
                Q_func_hat.load_state_dict(Q_func.state_dict())

        if done:
            trainer = env.train([None, next(op_agents)])
            obs = trainer.reset()
            state = process_state(obs, device)
        else:
            state = next_state
        
        ## validation
        if step % train_config.validate_every == 0:
            val_reward = validate(
                Q_func, 
                num_episodes=train_config.num_episodes,
                op_agent=train_config.val_op_agent, 
                device=device
            )
            logs['val_reward'].append(val_reward)
            logs['val_step'].append(step)

            print(f'| step: {step} | val_reward: {val_reward:.4f} |')
    
    return logs