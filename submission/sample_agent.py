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

            left_emptiness = sum(1 for x in range(BOARD_WIDTH) if column_heights[x] == 0)
            right_emptiness = sum(1 for x in range(BOARD_WIDTH) if column_heights[BOARD_WIDTH - x - 1] == 0)
            center_emptiness = sum(1 for x in range(BOARD_WIDTH) if column_heights[x] == 0 and x > 1 and x < BOARD_WIDTH - 2)

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
                left_emptiness,
                right_emptiness,
                center_emptiness
            ])
            
            score = np.dot(self.weights, feature_vector)
            
            if len(clearable_lines) >= 2:
                score = float('-inf')
            
            if not b.is_game_running():
                score = float('inf')

            return score
        
        return min(MOVES, key=score_moves)

SelectedAgent = HeuristicAgent

#####################################################################
#   This runs your agent and communicates with DOXA over stdio,     #
#   so please do not touch these lines unless you are comfortable   #
#   with how DOXA works, otherwise your agent may not run.          #
#####################################################################
if __name__ == "__main__":
    weights = [0.6161222852508982, 3683.333054527225, 61.86765240692307, -1.7965668217660635, -0.44346395803540994, 957.4515643158011, 2628.405761664556, -63.09247365268259, 0.01665550011904729]
    weights = [1059.6773122574548, 3688.507755374621, 63.99221519900783, -2.136861886361613, 8.648616279998466, 960.8049734265385, 2623.3197690456636, -63.38334297511657, 0.671813882739172, 1016.8431165331463, 1059.6773122574548, 924.4418737706224]
    weights = [240.35183629432146, -6.053133046881249, 63.81097752251967, -0.010673201605337187, 0.3486191627718447, 199.80586267932406, 1060.4195638828246, -7.867480710255641, 0.0015930841825072665, 121.70550775234003, 240.35183629432146, 1.2455805780899138]
    agent = SelectedAgent(np.array(weights))
    main(agent)