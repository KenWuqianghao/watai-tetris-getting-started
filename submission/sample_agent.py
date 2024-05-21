from random import randrange
from typing import Sequence, Union
import numpy as np

from tetris import Action, BaseAgent, Board, main
from tetris.constants import BOARD_HEIGHT, BOARD_WIDTH
from tetris.moves import MOVES

def calculate_heights(board):
    """
    :param board: the current game state
    :return: a list containing the height of each block column on the board
    """
    heights = []

    for x in range(BOARD_WIDTH):
        for y in range(BOARD_HEIGHT):
            if board[y][x] is not None:
                heights.append(BOARD_HEIGHT - y)
                break
        else:
            heights.append(0)

    return heights


def count_holes(board, heights):
    """
    :param board: the current game state
    :param heights: the height of each block column
    :return: the number of holes on the board
    """
    holes = 0
    for x, h in enumerate(heights):
        if h <= 1:
            continue

        for y in range(BOARD_HEIGHT - h, BOARD_HEIGHT):
            if board[y][x] is None:
                holes += 1

    return holes

class HeuristicAgent(BaseAgent):
    def __init__(self, weights) -> None:
        super().__init__()
        self.weights = weights

    async def play_move(self, board: Board) -> Union[Action, Sequence[Action]]:
        """Makes at least one Tetris move.
        
        If a sequence of moves is returned, they are made in order
        until the piece lands (with any remaining moves discarded).
        
        Args:
            board (Board): The Tetris board.
        
        Returns:
            Union[Action, Sequence[Action]]: The action(s) to perform.
        """

        def score_moves(m):
            """
            :param m: a potential sequence of moves
            :return: a metric that estimates the effectiveness of the input move sequence
            """

            # simulate the board after move sequence m
            b = board.with_moves(m)
            
            # number of potential lines to clear
            clearable_lines = b.find_lines_to_clear()

            # column heights
            column_heights = calculate_heights(b)

            # gaps under blocks
            hole_count = count_holes(b, column_heights)

            # spikiness and bumpiness
            spikiness = []
            bumpiness = 0
            for i in range(len(column_heights) - 1):
                height_diff = abs(column_heights[i] - column_heights[i + 1])
                spikiness.append(height_diff)
                bumpiness += height_diff
            total_spikiness = sum(spikiness)

            # aggregate height
            total_height = sum(column_heights)

            # maximum column height
            max_height = max(column_heights)

            # board weighted holes
            weighted_holes = sum(hole * (index + 1) for index, hole in enumerate(column_heights))
            
            total_blocks = sum(BOARD_WIDTH - line.count(None) for line in b)

            # feature vector
            feature_vector = np.array([
                -len(clearable_lines),
                sum(column_heights),
                hole_count,
                total_blocks,
                total_spikiness,
                bumpiness,
                total_height,
                max_height,
                weighted_holes,
            ])
            
            return np.dot(self.weights, feature_vector)

        return min(MOVES, key=score_moves)

SelectedAgent = HeuristicAgent

#####################################################################
#   This runs your agent and communicates with DOXA over stdio,     #
#   so please do not touch these lines unless you are comfortable   #
#   with how DOXA works, otherwise your agent may not run.          #
#####################################################################
if __name__ == "__main__":
    weights = [1070.558745981553, 963.7975031106122, 1380.587323768055, 74.34870576097629, -800.7595753505859, 1159.9820521194808, 503.7611272855837, -3.7377129147859023, 14.857216139137405]
    weights = [1282.499711876682, 1275.8145875674409, 1501.743576191133, 374.9599710995053, -750.3724423249874, 1358.3003212718686, 780.6393226684344, 115.75496461733854, -3.860261029035676]
    weights = [ 1.46550562, -0.66736624,  1.85619377,  1.37476826,  0.42216046, 0.01825443,  1.06798135,  0.47391155,  0.06139508]
    agent = SelectedAgent(np.array(weights))
    main(agent)