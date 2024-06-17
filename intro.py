#Attempt at making tic-tac-toe using a Q table -- failed due to a confusing and innacurate representation ofo the game that was difficult to work with and rushed
import random
import math
ttt = [[-1 for _ in range(3)] for _ in range(3)]
qTable = []
def printer():
    [print(t) for t in ttt]
def checkMove(x,y):
    if ttt[x][y] != -1:
        return False
    return True
def move(x,y,num):
    if checkMove(x,y):
        ttt[x][y] = num
def checkforwin(x,y):
    for c in [[1,0], [0,1],[1,1],[1,-1]]:
        if ttt[(x + c[0]) % 3][(y + c[1]) % 3] == ttt[(x - c[0]) % 3][(y - c[1]) % 3] == ttt[x][y] and ttt[x][y] != -1:
            return True
    return False
def validate(x,y, lost = False):
    if not checkMove(x,y) or lost:
        return -100
    if checkforwin(x,y):
        return 100
    return 0
def getAiMove():
    if ttt not in qTable:
        qTable.append([[tt[:] for tt in ttt], [random.uniform(-1,1) for _ in range(9)]])
    index = max([x if ttt in qTable[x] else -1 for x in range(len(qTable))])
    move = qTable[index][1].index(max(qTable[index][1]))
    r = (move % 3, math.floor(move/3.1))
    return move % 3, math.floor(move/3.1)
moves = []
lose = False
def learn():
    discount_factor = 0.8
    learning_rate = 0.1
    for i,m in enumerate(moves):
        state = m[0]
        action = m[1][0] + (m[1][1] * 3)
        reward = m[2]

        qIndex = max([x if state in qTable[x] else -1 for x in range(len(qTable))])
        qVal = qTable[qIndex][1][action]
        if i < len(moves)-1:
            next_qIndex = max([x if moves[i+1][0] in qTable[x] else -1 for x in range(len(qTable))])
            next_qVal = max(qTable[next_qIndex][1])
        else:
            next_qVal = 0
        #bellman equation
        newParam = qVal + (learning_rate * (reward + discount_factor * next_qVal - qVal))
        qTable[qIndex][1][action] = newParam
for _ in range(5000):
    ttt = [[-1 for _ in range(3)] for _ in range(3)]
    moves = []
    lose = False
    for i in range(9):
        playerMove = random.randint(0,9)#list(map(int, input("Move?").split(",")))[::-1]
        playerMove = playerMove % 3, math.floor(playerMove / 3.1)
        move(*playerMove,"O")
        if checkforwin(*playerMove):
            lose = True
            print("WIN!")
        print()
        aiMove = getAiMove()
        moves.append([[row[:] for row in ttt], aiMove, validate(*aiMove, lose)])
        move(*aiMove,"X")
        printer()
        if lose:
            break
        if checkforwin(*aiMove):
            print("WIN!")
            break
    learn()
    print("Learned!")
print(qTable)