#red guy touches green goal with neural net training
from neuralnetwork import *
from intro2 import Game
from time import sleep
from random import random, randint
#base network and target network
nn = Network(100, 16, 4, 2)
target = Network(100, 16, 4, 2)
#begin randomized
nn.randomize()
target.randomize()

discount_factor = 0.5
#learning rates
nn.lr = 0.01
target.lr = nn.lr

#takes 100 possible input locations, translate 2d position to 1d array (probbaly a numpy function that could do this better)
def translate_state(state):
    res = np.zeros(100)
    res[state[0] + 10 * state[1]] = 1
    return res
#ii represents amount of cycles run to update target network
def learn(moves, ii):
    if ii % 10 == 0:
        #update target network every 10 iterations
        for i in range(len(target.layers)):
            target.layers[i].weights = nn.layers[i].weights.copy()
            target.layers[i].bias = nn.layers[i].bias.copy()
    for i,move in enumerate(moves):
        state, action, reward = move
        #get future action from target network assuming not in terminal state
        try:
            future_action = max(target.forward_prop(translate_state(moves[i+1][0])))
        except IndexError:
            future_action = 0
        #bellman
        target_Q_value = reward + discount_factor * future_action
        #create expected output
        target_Q_set = nn.forward_prop(translate_state(state))
        target_Q_set[action] = target_Q_value
        #update network
        nn.apply_changes(nn.back_prop(translate_state(state), target_Q_set))

translate = [(0,1),(1,0),(0,-1),(-1,0)]
for i in range(1000):
    g = Game()
    print("training...", i)
    #run game
    while(True):
        position = np.array([g.x,g.y])
        #add randomness to cyclels that decrease as cycles continue
        if random() > 1 - i/500:
            nextMove = np.argmax(nn.forward_prop(translate_state(position)))
        else:
            nextMove = randint(0,3)
        g.move(*translate[nextMove])
        reward = g.evaluate()
        g.addToMoveset(position, nextMove, reward)
        #break in terminal state
        if reward <= -100 or reward >= 100:
            break
    learn(g.moves,i)
#simulate visually
while True:
    g = Game()
    print("training...", i)
    while(True):
        position = np.array([g.x,g.y])
        nextMove = np.argmax(nn.forward_prop(translate_state(position)))
        g.move(*translate[nextMove])
        reward = g.evaluate()
        g.addToMoveset(position, nextMove, reward)
        if reward <= -100 or reward >= 100:
            break
        g.update()
    learn(g.moves,i)