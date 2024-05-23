import sys
import warnings
warnings.filterwarnings("ignore")

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
    r, s, t, u, v, w, x, y, z, l, r, c = params['r'], params['s'], params['t'], params['u'], params['v'], params['w'], params['x'], params['y'], params['z'], params['l'], params['r'], params['c']
    weights = np.array([r, s, t, u, v, w, x, y, z, l, r, c])
    agent = SelectedAgent(weights)
    scores = []
    for seed in eval_seeds:
        game = Game(agent, seed=int(seed))
        async for item in game.run():
            pass
        scores.append(game.score)
    min_score = -min(scores)
    return min_score

def objective(params):
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(get_score(params))

async def calculate_score(params, seed):
    r, s, t, u, v, w, x, y, z, l, r, c = params['r'], params['s'], params['t'], params['u'], params['v'], params['w'], params['x'], params['y'], params['z'], params['l'], params['r'], params['c']
    weights = np.array([r, s, t, u, v, w, x, y, z, l, r, c])
    agent = SelectedAgent(weights)
    scores = []
    game = Game(agent, seed=seed)
    async for item in game.run():
        pass
    scores.append(game.score)
    min_score = -min(scores)
    return min_score

def get_eval_seeds(params, seed):
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(calculate_score(params, seed))

eval_seeds = np.random.randn(10)

space = {
    'r': hp.normal('r', 0.6, 10.7),
    's': hp.normal('s', 3683, 9.6),
    't': hp.normal('t', 61, 13.8),
    'u': hp.normal('u', -1.7, 0.7),
    'v': hp.normal('v', -0.4, 8),
    'w': hp.normal('w', 957, 11.6),
    'x': hp.normal('x', 2628, 5),
    'y': hp.normal('y', -63, 0.3),
    'z': hp.normal('z', 0.01, 1.4),
    'l': hp.normal('l', 1000, 100),
    'r': hp.normal('r', 1000, 100),
    'c': hp.normal('c', 1000, 100),
}

while True:
    best = fmin(
        fn=objective, # Objective Function to optimize
        space=space, # Hyperparameter's Search Space
        algo=tpe.suggest, # Optimization algorithm (representative TPE)
        max_evals=50 # Number of optimization attempts
    )

    best_weights = [best['r'], best['s'], best['t'], best['u'], best['v'], best['w'], best['x'], best['y'], best['z'], best['l'], best['r'], best['c']]
    print(best_weights)

    space = {
        'r': hp.normal('r', best['r'], abs(best['r'] * 0.1)),
        's': hp.normal('s', best['s'], abs(best['s'] * 0.1)),
        't': hp.normal('t', best['t'], abs(best['t'] * 0.1)),
        'u': hp.normal('u', best['u'], abs(best['u'] * 0.1)),
        'v': hp.normal('v', best['v'], abs(best['v'] * 0.1)),
        'w': hp.normal('w', best['w'], abs(best['w'] * 0.1)),
        'x': hp.normal('x', best['x'], abs(best['x'] * 0.1)),
        'y': hp.normal('y', best['y'], abs(best['y'] * 0.1)),
        'z': hp.normal('z', best['z'], abs(best['z'] * 0.1)),
        'l': hp.normal('l', best['l'], abs(best['l'] * 0.1)),
        'r': hp.normal('r', best['r'], abs(best['r'] * 0.1)),
        'c': hp.normal('c', best['c'], abs(best['c'] * 0.1)),
    }

    scores = {}

    for i in range(25):
        seed = random.randint(0, 1000)
        scores[seed] = get_eval_seeds(best, seed)   

    # find the 3 worst seeds of scores dictionary
    worst_seeds = sorted(scores, key=scores.get, reverse=True)[:3]

    # find the 3 best seeds of scores dictionary
    best_seeds = sorted(scores, key=scores.get, reverse=False)[:3]

    # find the 4 middle seeds of scores dictionary
    middle_seeds = sorted(scores, key=scores.get, reverse=False)[14:17]

    eval_seeds = np.concatenate([best_seeds, worst_seeds, middle_seeds])