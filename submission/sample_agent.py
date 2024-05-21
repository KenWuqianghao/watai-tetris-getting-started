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

            # spikiness
            spikiness = [abs(x - y) for x, y in zip(column_heights, column_heights[1:])]
            total_spikiness = sum(spikiness)

            # total number of blocks
            total_blocks = sum(BOARD_WIDTH - line.count(None) for line in b)

            feature_vector = np.array(
                [
                    -len(clearable_lines),
                    sum(column_heights),
                    hole_count,
                    total_blocks,
                    total_spikiness,
                ]
            )
            
            return np.dot(self.weights, feature_vector)

        return min(MOVES, key=score_moves)


SelectedAgent = HeuristicAgent

#####################################################################
#   This runs your agent and communicates with DOXA over stdio,     #
#   so please do not touch these lines unless you are comfortable   #
#   with how DOXA works, otherwise your agent may not run.          #
#####################################################################

if __name__ == "__main__":
    w = [7.551385770181472, -2.7139623796583905, 6.3570363173895466, 7.950006016113665, 2.366147437990671]
    agent = SelectedAgent(np.array(w))
    main(agent)