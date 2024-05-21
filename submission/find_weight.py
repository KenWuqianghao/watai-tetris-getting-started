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

eval_seeds = np.random.randint(100, size=(10))

async def get_score(params):
    r, s, t, u, v, w, x, y, z = params['r'], params['s'], params['t'], params['u'], params['v'], params['w'], params['x'], params['y'], params['z']
    weights = np.array([r, s, t, u, v, w, x, y, z])
    agent = SelectedAgent(weights)
    scores = []
    for seed in eval_seeds:
        game = Game(agent, seed=seed)
        async for item in game.run():
            pass
        scores.append(game.score)
    min_score = -min(scores)
    return min_score

def objective(params):
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(get_score(params))

space = {
    'r': hp.normal('r', 1070, 100),
    's': hp.normal('s', 963, 100),
    't': hp.normal('t', 1380, 100),
    'u': hp.normal('u', 74, 100),
    'v': hp.normal('v', -800, 100),
    'w': hp.normal('w', 1159, 100),
    'x': hp.normal('x', 503, 100),
    'y': hp.normal('y', -3, 100),
    'z': hp.normal('z', 14, 100),
}

while True:
    best = fmin(
        fn=objective, # Objective Function to optimize
        space=space, # Hyperparameter's Search Space
        algo=tpe.suggest, # Optimization algorithm (representative TPE)
        max_evals=1 # Number of optimization attempts
    )

    best_weights = [best['r'], best['s'], best['t'], best['u'], best['v'], best['w'], best['x'], best['y'], best['z']]
    print(best_weights)

    space = {
        'r': hp.normal('r', best['r'], best['r'] / 10),
        's': hp.normal('s', best['s'], best['s'] / 10),
        't': hp.normal('t', best['t'], best['t'] / 10),
        'u': hp.normal('u', best['u'], best['u'] / 10),
        'v': hp.normal('v', best['v'], best['v'] / 10),
        'w': hp.normal('w', best['w'], best['w'] / 10),
        'x': hp.normal('x', best['x'], best['x'] / 10),
        'y': hp.normal('y', best['y'], best['y'] / 10),
        'z': hp.normal('z', best['z'], best['z'] / 10),
    }

    scores = {}
    for i in range(100):
        seed = random.randint(0, 1000000)
        eval_seeds = np.array([seed])
        scores[seed] = objective(best_weights)

    # find the 3 worst seeds of scores dictionary
    worst_seeds = sorted(scores, key=scores.get, reverse=True)[:3]

    # find the 3 best seeds of scores dictionary
    best_seeds = sorted(scores, key=scores.get, reverse=False)[:3]

    # find the 4 middle seeds of scores dictionary
    middle_seeds = sorted(scores, key=scores.get, reverse=False)[48:52]

    eval_seeds = np.concatenate([best_seeds, worst_seeds, middle_seeds])