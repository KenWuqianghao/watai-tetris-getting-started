import sys

# EDIT THIS
sys.path.append('/Users/kenwu/Documents/Github/watai-tetris-getting-started/')

from submission.sample_agent import SelectedAgent
from submission.tetris.game import Game

from tqdm import tqdm
import numpy as np

# Import HyperOpt Library
from hyperopt import tpe, hp, fmin
import numpy as np

import random
import asyncio

async def get_score(params):
    v, w, x, y, z = params['v'], params['w'], params['x'], params['y'], params['z']
    weights = np.array([v, w, x, y, z])
    agent = SelectedAgent(weights)
    scores_list = []
    for trial in range(100):
        GAME_SEED = random.randint(0, 1000)
        game = Game(agent, seed=GAME_SEED)
        async for item in game.run():
            pass
        scores = {}
        scores.setdefault(game.score, []).append(w)
        max_score = max(scores.keys())
        scores_list.append(max_score)
    average_score = np.mean(scores_list)
    return -average_score

def objective(params):
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(get_score(params))

space = {
    'v': hp.uniform('v', -10, 10),
    'w': hp.uniform('w', -10, 10),
    'x': hp.uniform('x', -10, 10),
    'y': hp.uniform('y', -10, 10),
    'z': hp.uniform('z', -10, 10)
}

best = fmin(
    fn=objective, # Objective Function to optimize
    space=space, # Hyperparameter's Search Space
    algo=tpe.suggest, # Optimization algorithm (representative TPE)
    max_evals=3000 # Number of optimization attempts
)

best_weights = [best['v'], best['w'], best['x'], best['y'], best['z']]
print(best_weights)