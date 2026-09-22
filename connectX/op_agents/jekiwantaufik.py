
import math
import random

def jekiwantaufik(obs, conf):

    ROWS = conf.rows
    COLS = conf.columns
    INAROW = conf.inarow

    PLAYER = obs.mark
    OPPONENT = 1 if PLAYER == 2 else 2

    board = [
        obs.board[i * COLS:(i + 1) * COLS]
        for i in range(ROWS)
    ]

    # ---------------------------------------------------
    # Transposition table
    # ---------------------------------------------------

    cache = {}

    # ---------------------------------------------------
    # Valid moves
    # ---------------------------------------------------

    def valid_moves(board):

        return [
            c for c in range(COLS)
            if board[0][c] == 0
        ]

    # ---------------------------------------------------
    # Center-first move ordering
    # ---------------------------------------------------

    def ordered_moves(board):

        valid = valid_moves(board)

        center = COLS // 2

        return sorted(
            valid,
            key=lambda c: abs(center - c)
        )

    # ---------------------------------------------------
    # Dynamic search depth
    # ---------------------------------------------------

    def get_search_depth(board):

        moves_played = sum(
            cell != 0
            for row in board
            for cell in row
        )

        # Early game
        if moves_played < 10:
            return 4

        # Mid game
        elif moves_played < 20:
            return 5

        # Late game
        else:
            return 6

    # ---------------------------------------------------
    # Find next open row
    # ---------------------------------------------------

    def get_next_open_row(board, col):

        for r in range(ROWS - 1, -1, -1):

            if board[r][col] == 0:
                return r

    # ---------------------------------------------------
    # Simulate move
    # ---------------------------------------------------

    def drop_piece(board, row, col, piece):

        temp = [r[:] for r in board]

        temp[row][col] = piece

        return temp

    # ---------------------------------------------------
    # Check dangerous move
    # ---------------------------------------------------

    def is_safe_move(board, col, piece):

        row = get_next_open_row(
            board,
            col
        )

        temp = drop_piece(
            board,
            row,
            col,
            piece
        )

        opponent = (
            PLAYER
            if piece == OPPONENT
            else OPPONENT
        )

        for opp_col in valid_moves(temp):

            opp_row = get_next_open_row(
                temp,
                opp_col
            )

            opp_temp = drop_piece(
                temp,
                opp_row,
                opp_col,
                opponent
            )

            if winning_move(
                opp_temp,
                opponent
            ):
                return False

        return True

    # ---------------------------------------------------
    # Count winning threats
    # ---------------------------------------------------

    def count_winning_moves(board, piece):

        count = 0

        for col in valid_moves(board):

            row = get_next_open_row(
                board,
                col
            )

            temp = drop_piece(
                board,
                row,
                col,
                piece
            )

            if winning_move(temp, piece):
                count += 1

        return count

    # ---------------------------------------------------
    # Detect opponent fork threats
    # ---------------------------------------------------

    def opponent_fork_exists(board):

        for col in valid_moves(board):

            row = get_next_open_row(
                board,
                col
            )

            temp = drop_piece(
                board,
                row,
                col,
                OPPONENT
            )

            threats = count_winning_moves(
                temp,
                OPPONENT
            )

            if threats >= 2:
                return True

        return False

    # ---------------------------------------------------
    # Hashable board key
    # ---------------------------------------------------

    def board_key(board):

        return tuple(
            tuple(row)
            for row in board
        )

    # ---------------------------------------------------
    # Win detection
    # ---------------------------------------------------

    def winning_move(board, piece):

        # Horizontal
        for r in range(ROWS):

            for c in range(COLS - 3):

                if all(
                    board[r][c + i] == piece
                    for i in range(4)
                ):
                    return True

        # Vertical
        for r in range(ROWS - 3):

            for c in range(COLS):

                if all(
                    board[r + i][c] == piece
                    for i in range(4)
                ):
                    return True

        # Positive diagonal
        for r in range(ROWS - 3):

            for c in range(COLS - 3):

                if all(
                    board[r + i][c + i] == piece
                    for i in range(4)
                ):
                    return True

        # Negative diagonal
        for r in range(3, ROWS):

            for c in range(COLS - 3):

                if all(
                    board[r - i][c + i] == piece
                    for i in range(4)
                ):
                    return True

        return False

    # ---------------------------------------------------
    # Evaluate 4-cell window
    # ---------------------------------------------------

    def evaluate_window(window, piece):

        score = 0

        opp_piece = (
            PLAYER
            if piece == OPPONENT
            else OPPONENT
        )

        # Winning move
        if window.count(piece) == 4:
            score += 100

        # Strong attack
        elif (
            window.count(piece) == 3
            and window.count(0) == 1
        ):
            score += 10

        # Weak attack
        elif (
            window.count(piece) == 2
            and window.count(0) == 2
        ):
            score += 5

        # Strong defense
        if (
            window.count(opp_piece) == 3
            and window.count(0) == 1
        ):
            score -= 80

        return score

    # ---------------------------------------------------
    # Board evaluation
    # ---------------------------------------------------

    def score_position(board, piece):

        score = 0

        # Center preference
        center = [
            board[r][COLS // 2]
            for r in range(ROWS)
        ]

        score += center.count(piece) * 6

        # Horizontal
        for r in range(ROWS):

            row = board[r]

            for c in range(COLS - 3):

                window = row[c:c + 4]

                score += evaluate_window(
                    window,
                    piece
                )

        # Vertical
        for c in range(COLS):

            col = [
                board[r][c]
                for r in range(ROWS)
            ]

            for r in range(ROWS - 3):

                window = col[r:r + 4]

                score += evaluate_window(
                    window,
                    piece
                )

        # Positive diagonal
        for r in range(ROWS - 3):

            for c in range(COLS - 3):

                window = [
                    board[r + i][c + i]
                    for i in range(4)
                ]

                score += evaluate_window(
                    window,
                    piece
                )

        # Negative diagonal
        for r in range(3, ROWS):

            for c in range(COLS - 3):

                window = [
                    board[r - i][c + i]
                    for i in range(4)
                ]

                score += evaluate_window(
                    window,
                    piece
                )

        return score

    # ---------------------------------------------------
    # Minimax + Alpha Beta
    # ---------------------------------------------------

    def minimax(
        board,
        depth,
        alpha,
        beta,
        maximizing
    ):

        valid = [

            col for col in ordered_moves(board)

            if is_safe_move(
                board,
                col,
                PLAYER if maximizing else OPPONENT
            )
        ]

        # fallback
        if len(valid) == 0:
            valid = ordered_moves(board)

        # ---------------------------------------------------
        # Cache lookup
        # ---------------------------------------------------

        key = (
            board_key(board),
            depth,
            maximizing
        )

        if key in cache:
            return cache[key]

        terminal = (
            winning_move(board, PLAYER)
            or winning_move(board, OPPONENT)
            or len(valid) == 0
        )

        # ---------------------------------------------------
        # Terminal state
        # ---------------------------------------------------

        if depth == 0 or terminal:

            if winning_move(board, PLAYER):
                return (None, 1000000)

            elif winning_move(board, OPPONENT):
                return (None, -1000000)

            else:
                return (
                    None,
                    score_position(board, PLAYER)
                )

        # ---------------------------------------------------
        # Maximizing player
        # ---------------------------------------------------

        if maximizing:

            value = -math.inf

            best_col = random.choice(valid)

            for col in valid:

                row = get_next_open_row(
                    board,
                    col
                )

                temp = drop_piece(
                    board,
                    row,
                    col,
                    PLAYER
                )

                _, score = minimax(
                    temp,
                    depth - 1,
                    alpha,
                    beta,
                    False
                )

                if score > value:

                    value = score
                    best_col = col

                alpha = max(alpha, value)

                # Alpha-beta pruning
                if alpha >= beta:
                    break

            cache[key] = (best_col, value)

            return best_col, value

        # ---------------------------------------------------
        # Minimizing player
        # ---------------------------------------------------

        else:

            value = math.inf

            best_col = random.choice(valid)

            for col in valid:

                row = get_next_open_row(
                    board,
                    col
                )

                temp = drop_piece(
                    board,
                    row,
                    col,
                    OPPONENT
                )

                _, score = minimax(
                    temp,
                    depth - 1,
                    alpha,
                    beta,
                    True
                )

                if score < value:

                    value = score
                    best_col = col

                beta = min(beta, value)

                # Alpha-beta pruning
                if alpha >= beta:
                    break

            cache[key] = (best_col, value)

            return best_col, value

    # ---------------------------------------------------
    # Opening book strategy
    # ---------------------------------------------------

    moves_played = sum(
        cell != 0
        for row in board
        for cell in row
    )

    center_col = COLS // 2

    # First move:
    # always take center

    if moves_played == 0:

        return center_col

    # Second move:
    # prefer center again if possible

    if moves_played == 1:

        if center_col in valid_moves(board):
            return center_col

    # Early game:
    # strongly prefer center-area moves

    if moves_played < 6:

        center_order = sorted(
            valid_moves(board),
            key=lambda c: abs(center_col - c)
        )

        for col in center_order:

            if is_safe_move(
                board,
                col,
                PLAYER
            ):
                return col

    # ---------------------------------------------------
    # Immediate winning move
    # ---------------------------------------------------

    for col in ordered_moves(board):

        row = get_next_open_row(
            board,
            col
        )

        temp = drop_piece(
            board,
            row,
            col,
            PLAYER
        )

        if winning_move(temp, PLAYER):
            return col

    # ---------------------------------------------------
    # Immediate opponent block
    # ---------------------------------------------------

    for col in ordered_moves(board):

        row = get_next_open_row(
            board,
            col
        )

        temp = drop_piece(
            board,
            row,
            col,
            OPPONENT
        )

        if winning_move(temp, OPPONENT):
            return col

    # ---------------------------------------------------
    # Create double threat (fork)
    # ---------------------------------------------------

    for col in ordered_moves(board):

        row = get_next_open_row(
            board,
            col
        )

        temp = drop_piece(
            board,
            row,
            col,
            PLAYER
        )

        winning_threats = count_winning_moves(
            temp,
            PLAYER
        )

        if winning_threats >= 2:
            return col

    # ---------------------------------------------------
    # Prevent opponent fork
    # ---------------------------------------------------

    for col in ordered_moves(board):

        row = get_next_open_row(
            board,
            col
        )

        temp = drop_piece(
            board,
            row,
            col,
            PLAYER
        )

        if not opponent_fork_exists(temp):

            if is_safe_move(
                board,
                col,
                PLAYER
            ):
                return col

    # ---------------------------------------------------
    # Dynamic depth selection
    # ---------------------------------------------------

    max_depth = get_search_depth(board)

    # ---------------------------------------------------
    # Iterative deepening search
    # ---------------------------------------------------

    best_col = random.choice(
        valid_moves(board)
    )

    for depth in range(1, max_depth + 1):

        col, _ = minimax(
            board,
            depth=depth,
            alpha=-math.inf,
            beta=math.inf,
            maximizing=True
        )

        if col is not None:
            best_col = col

    return best_col