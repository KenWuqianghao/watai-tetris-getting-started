import random
from deap import base, creator, tools, algorithms
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

seeds = [741, 256, 674, 216, 288, 127, 797, 719, 489, 740, 768, 158, 458, 621, 755, 924, 734, 802,  64,  89]

async def get_score(agent):
    scores_list = []
    for i in range(3):
        GAME_SEED = random.randint(0, 1000)
        game = Game(agent, seed=GAME_SEED)
        async for item in game.run():
            pass
        scores = []
        scores.append(game.score)
    average_score = np.mean(scores)
    return average_score

def objective(agent):
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(get_score(agent))

# Define the fitness function
def evalTetris(individual):
    agent = SelectedAgent(np.array(individual))
    return (objective(agent),)  # simulate_game needs to be implemented

# Set up the genetic algorithm
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, -1.0, 1.0)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=9)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evalTetris)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

population = toolbox.population(n=100)
final_population = algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=100, verbose=True)
print(final_population)
best_weights = final_population[0]
print(best_weights)
# use pickle to save final_population
import pickle
with open('final_population.pkl', 'wb') as f:
    pickle.dump(final_population, f)