import numpy as np
import re


def validate_sokoban_solution(game_map, moves):
    """
    Validate if the given moves solve a single-box Sokoban puzzle.

    Parameters
    ----------
    game_map : np.ndarray
        2D numpy array with:
    moves : list of str
        Sequence of moves: 'left', 'right', 'up', 'down'

    Returns
    -------
    bool
        True if the moves lead to the box on the goal, False otherwise
    """
    # Move directions
    directions = {
        "up": (-1, 0),
        "down": (1, 0),
        "left": (0, -1),
        "right": (0, 1),
    }

    # Find positions
    player_pos = tuple(np.argwhere(game_map == 5)[0])
    box_pos = tuple(np.argwhere(game_map == 2)[0])
    goal_pos = tuple(np.argwhere(game_map == 3)[0])

    # Simulate moves
    for move in moves:
        if move not in directions:
            return 0

        d = directions[move]
        next_player = (player_pos[0] + d[0], player_pos[1] + d[1])

        # Check if player hits a wall
        if game_map[next_player] == 1:
            return False

        # If player moves into the box
        if next_player == box_pos:
            next_box = (box_pos[0] + d[0], box_pos[1] + d[1])
            # Check if box is blocked
            if (not (0 <= next_box[0] < game_map.shape[0] and 0 <= next_box[1] < game_map.shape[1])) \
               or game_map[next_box] == 1:
                return False
            # Move box
            box_pos = next_box

        # Move player
        player_pos = next_player

    # Check if solved
    return int(box_pos == goal_pos)


def extract_bbox(s: str):
    """
    Extract the contents inside //bbox{...}> from a string,
    split by commas, and strip whitespace.

    Args:
        s (str): Input string.

    Returns:
        list[str]: List of cleaned bbox values.
    """
    match = re.search(r"\\bbox\{([^}]*)\}", s)
    if not match:
        return []

    contents = match.group(1)
    return [x.strip() for x in contents.split(",")]


def sokoban_compute_reward(response_str, ground_truth, board):

    ans = extract_bbox(response_str)
    if len(ans) == 0:
        return 0
    ans = [item.strip().lower() for item in ans]
    board = np.stack(board)

    return validate_sokoban_solution(board, ans)
