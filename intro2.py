#get the red guy to hit the green guy
from numpy import argmax
from math import dist
from random import random, randint
from graphics import *
from time import sleep
win = GraphWin("THE GRAPH", 800,800, False)
win.setCoords(0,100,100,0)
grid = [[Rectangle(Point(y * 10,x * 10), Point(y * 10 +10, x * 10+10)) for x in range(10)]for y in range(10)]
for row in grid:
    for g in row:
        g.setFill("gray")
        g.draw(win)
        update()
class QTable:
    def __init__(self, game = None):
        self.table = {(i,j):[0 for _ in range(4)] for i in range(10) for j in range(10)}
        self.game = game
        #translator from move index to movement in 2d space
        self.translate = [(0,1),(1,0),(0,-1),(-1,0)]
        #how often it makes a random move
        self.variability = 0.5
    #get highest q value index for a specific state
    def getBest(self,x,y):
        if random() < self.variability:
            return randint(0,3)
        return argmax(self.table[(x,y)])
    #get highest q value for a specific state 
    def getBestValue(self,x,y):
        return max(self.table[(x,y)])
    #for each move, use bellman equation to improve q table
    def learn(self, moves):
        learning_rate = 0.1
        discount_factor = 0.9
        for i,m in enumerate(moves):
            state = m[0]
            action = m[1]
            reward = m[2]
            try:
                future_state = moves[i+1][0]
                future_action = self.getBestValue(*future_state)
            except IndexError:
                future_action = 0
            self.table[state][action] += learning_rate * (reward + (discount_factor * (future_action) - self.table[state][action]))

class Game:
    def __init__(self, goal = (9,9), maxTurns = 20):
        self.x,self.y = randint(0,9), randint(0,9)
        self.turns = 0
        self.moves = []
        self.goal = goal
        self.maxTurns = maxTurns
    #update visual grid
    def update(self):
        [[g.setFill("gray") for g in m] for m in grid]
        grid[self.goal[0]][self.goal[1]].setFill("green")
        grid[self.x][self.y].setFill("red")
        update() #graphic.py update function
    #update agent position
    def move(self, dx=0,dy=0):
        self.x+=dx
        self.y+=dy
        self.turns+=1
    def addToMoveset(self, pos, move, reward):
        self.moves.append((pos,move,reward))
    #give reward based on move
    def evaluate(self): 
        if self.turns > self.maxTurns or self.x < 0 or self.x > 9 or self.y < 0 or self.y > 9:
            return -100
        if (self.x,self.y) == self.goal:
            return 100 + self.maxTurns - self.turns
        #return distance to goal
        return dist(self.goal, (self.x,self.y))
    
goalPos = (9,4)
q = QTable()
def train(amt = 50000):
    for i in range(amt):
        print("training...",i)
        #slowly tune down variability with training
        q.variability = 0.5 - (0.5 * (i/amt))
        g = Game(goalPos)
        #run game
        while(True):
            position = (g.x,g.y)
            nextMove = q.getBest(*position)
            g.move(*q.translate[nextMove])
            reward = g.evaluate()
            g.addToMoveset(position, nextMove, reward)
            if reward <= -100 or reward >= 100:
                break
        q.learn(g.moves)
#show game to visualize progress
def sim():
    while True:
        g = Game(goalPos)
        while(True):
            position = (g.x,g.y)
            nextMove = q.getBest(*position)
            g.move(*q.translate[nextMove])
            reward = g.evaluate()
            g.addToMoveset(position, nextMove, reward)
            if reward <= -100 or reward >= 100:
                break
            g.update()
            #slow down game speed
            sleep(0.1)
