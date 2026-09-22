import torch


def estimate_state(state, env_config, train_config, device):
    rows = env_config.rows
    columns = env_config.columns
    inarow = env_config.inarow

    state = state.view(rows, columns)


    def count_inarow(row):    
        total_inarow = torch.zeros(inarow + 1, device=device, dtype=torch.int64)
        cur_inarow = torch.zeros(inarow + 1, device=device, dtype=torch.int64)
        num_empty = 0
        num_inarow = 0
        
        EMPTY = 0
        OPPONENT = -1
        MARK = 1

        last_cell = OPPONENT


        def process_end_of_inarow(total_inarow, cur_inarow):
            if last_cell == MARK:
                cur_inarow[num_inarow] += 1

            # validate inarow
            weights = torch.arange(inarow + 1, device=device)
            if (cur_inarow * weights).sum().item() + num_empty >= inarow:
                total_inarow += cur_inarow

            return total_inarow, cur_inarow


        for cell in row:
            if cell == MARK:
                num_inarow += 1

            if cell == EMPTY:
                num_empty += 1
                if last_cell == MARK:
                    cur_inarow[num_inarow] += 1
                    num_inarow = 0
                
            if cell == OPPONENT:
                total_inarow, cur_inarow = process_end_of_inarow(total_inarow, cur_inarow)
                cur_inarow = torch.zeros(inarow + 1, device=device, dtype=torch.int64)
                num_inarow = 0
                num_empty = 0

            last_cell = cell

        total_inarow, cur_inarow = process_end_of_inarow(total_inarow, cur_inarow)
            
        return total_inarow

    total_inarow = torch.zeros(inarow + 1, device=device)

    for _state in [state, state.T]:
        for row in _state:
            total_inarow += count_inarow(row)

    for _state in [state, state.fliplr()]:
        for offset in range(-1 * (rows - inarow), columns - inarow + 1):
            row = _state.diag(offset)
            total_inarow += count_inarow(row)

    return total_inarow * train_config.reward_for_inarow


def compute_reward_strict(reward, state, next_state, env_config, train_config, device):
    if reward is None:
        return train_config.reward_for_None
    
    state_value = estimate_state(state, env_config, train_config, device)
    next_state_value = estimate_state(next_state, env_config, train_config, device)

    return reward + (train_config.state_reward_coeff * (next_state_value - state_value)).sum().item()


def compute_reward_soft(reward, train_config):
    return train_config.reward_for_None if reward is None else reward


# def compute_reward(reward, state, action, env_config, train_config):
#     if reward is None:
#         return train_config.reward_for_None

#     rows = env_config.rows
#     columns = env_config.columns
#     inarow = env_config.inarow

#     state = state.view(rows, columns)
    
#     def count_inarow(dx, dy, mark):
#         cur_inarow = 1
#         x = action + dx
#         y = (state[:,action]==0).sum() + dy

#         while 0 <= x < columns and 0 <= y < rows and state[y,x] == mark: 
#             cur_inarow += 1
#             x += dx
#             y += dy

#         return cur_inarow
    
#     def valid_inarow(dx, dy):
#         cur_inarow = count_inarow(dx, dy, mark=1)
#         empty_inarow = count_inarow(-dx, -dy, mark=0)

#         if empty_inarow - 1 >= inarow - cur_inarow:
#             return cur_inarow
#         return 0
    
#     for dx, dy in [(-1, 0), (1, 0), (-1, 1), (1, 1), (0, 1), (-1, 1), (1, 1)]:
#         reward += train_config.reward_for_inarow[valid_inarow(dx, dy)]
    
#     return reward


# print(inarow)
# state = torch.tensor([
#     [ 0,  0,  0,  0,  0,  0,  0],
#     [ 0,  0,  0,  0,  0,  0,  0],
#     [ 0,  0,  0,  0, -1,  0,  0],
#     [ 0,  0,  0,  1,  1,  0,  0],
#     [ 0,  0,  0, -1, -1,  0,  0],
#     [ 1,  0, -1,  1,  1, -1,  0]], device=device)

# next_state = torch.tensor([
#     [ 0,  0,  0,  0,  0,  0,  0],
#     [ 0,  0,  0,  0,  0,  0,  0],
#     [ 0,  0,  0,  0, -1,  0,  0],
#     [ 0,  0,  0,  1,  1,  0,  0],
#     [ 0,  0,  1, -1, -1,  0,  0],
#     [ 1,  -1, -1,  1,  1, -1,  0]], device=device)

# # print(estimate_state(state, env_config, train_config, device))
# print(compute_reward(0, state, next_state, env_config, train_config, device))