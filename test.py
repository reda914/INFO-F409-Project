import numpy as np
import random


pairings = []
players = [10,20,30,40,50,60]
for _ in range(3):
    index = random.randrange(len(players))
    elem1 = players.pop(index)

    index = random.randrange(len(players))
    elem2 = players.pop(index)

    pairings.append((elem1, elem2))

print(pairings)
print(pairings[0][0])