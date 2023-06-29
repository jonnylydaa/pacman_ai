from pacai.util import reflection
from pacai.agents.capture.reflex import ReflexCaptureAgent
from pacai.agents.capture.capture import CaptureAgent
from pacai.util import probability
from pacai.core.directions import Directions
import random

def createTeam(firstIndex, secondIndex, isRed,
        first = 'pacai.student.myTeam.OffensiveQAgent',
        second = 'pacai.student.myTeam.DefensiveQAgent'):
    """
    This function should return a list of two agents that will form the capture team,
    initialized using firstIndex and secondIndex as their agent indexed.
    isRed is True if the red team is being created,
    and will be False if the blue team is being created.
    """

    firstAgent = OffensiveQAgent
    secondAgent = DefensiveQAgent

    return [
        firstAgent(firstIndex),
        secondAgent(secondIndex),
    ]
agent1Weights = {
            'successorScore': -576,
            'distanceToFood': 150,
            'ateFood' : 1500,
            'ghostDistance' : -851
                }

agent2Weights = {
            'onDefense': 543,
            'invaderDistance': -176.24,
            'captured' : 499,
            'foeDistance' : -94.4
        }

#agent1Weights = {}
#agent2Weights = {}
class QCap(ReflexCaptureAgent):
    def __init__  (self, index, timeComputing = 0.1, **kwargs):
        super().__init__(index, **kwargs)
        self.weights = agent2Weights
        if index == 0 or index == 1:
            self.weights = agent1Weights
        self.discountRate = 0.9
        self.epsilon = 0
        self.alpha = 0.2

    def getValue(self, state):
        """
        Return the value of the best action in a state.
        I.E., the value of the action that solves: `max_action Q(state, action)`.
        Where the max is over legal actions.
        Note that if there are no legal actions, which is the case at the terminal state,
        you should return a value of 0.0.

        This method pairs with `QLearningAgent.getPolicy`,
        which returns the actual best action.
        Whereas this method returns the value of the best action.
        """
        # if terminal state, return 0 to avoid returning -9999999
        if state == "TERMINAL_STATE":
            return 0
        maxVal = -99999999
        # find Q values of each possible action, choose best
        for act in state.getLegalActions(self.index):
            if self.getQValue(state, act) > maxVal:
                maxVal = self.getQValue(state, act)
        # prevents returning -999999 upon ending game
        if maxVal == -99999999:
            maxVal = 0
        return maxVal

    def getQValue(self, state, action):
        featureVector = self.getFeatures(state, action)
        qVal = 0
        for i in featureVector:
            # update weights if feature not seen
            if i not in self.weights:
                self.weights.update({i: 0})
            # add weight * feat to qVal
            qVal += featureVector[i] * self.weights[i]
        return qVal

    def update(self, state, action, nextState, reward):
        correction = (reward + (self.discountRate * self.getValue(nextState)))
        correction -= self.getQValue(state, action)
        #print(correction)
        # update feats in weight vector according to feats extracted from the state
        for i in self.getFeatures(state, action):
            feats = self.getFeatures(state, action)[i]
            self.weights[i] += self.alpha * correction * feats
        #print(self.weights)

    def getPolicy(self, state):
        """
        Return the best action in a state.
        I.E., the action that solves: `max_action Q(state, action)`.
        Where the max is over legal actions.
        Note that if there are no legal actions, which is the case at the terminal state,
        you should return a value of None.

        This method pairs with `QLearningAgent.getValue`,
        which returns the value of the best action.
        Whereas this method returns the best action itself.
        """
        # use list for actions to make random selection easier
        maxVal = float('-inf')
        maxAct = list()
        for act in state.getLegalActions(self.index):
            # go through actions, pick max q value and update maxAct accordingly
            if self.getQValue(state, act) > maxVal:
                maxVal = self.getQValue(state, act)
                maxAct = [act]
            # if tied, add action to maxAct so random.choice can be used
            elif self.getQValue(state, act) == maxVal:
                maxAct.append(act)
        if len(maxAct) > 0:
            return random.choice(maxAct)
        return None
        
    def getFeatures(self, gameState, action):
        pass

    def chooseAction(self, state):
        # use epsilon and flipCoin to randomly choose between policy and random action
        if probability.flipCoin(1 - self.epsilon):
            act = self.getPolicy(state)
            self.update(state, act, state.generateSuccessor(self.index, act), self.getReward(state, act))
            return act
        act = random.choice(state.getLegalActions(self.index))
        self.update(state, act, state.generateSuccessor(self.index, act), self.getReward(state, act))
        return act

class DefensiveQAgent(QCap):

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)
        
        #foe is the opponent's offensive agent
        self.foe = -1

    def getFeatures(self, gameState, action):
        features = {}
        successor = self.getSuccessor(gameState, action)
        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        features['bias'] = 1

        # Computes whether we're on defense (1) or offense (0).
        features['onDefense'] = 1
        if (myState.isPacman()):
            features['onDefense'] = 0

        # Computes distance to invaders we can see.
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman() and a.getPosition() is not None]
        
        #Find offensive agent and mark it so defense agent can mirror its movements
        for enemy in self.getOpponents(successor):
            if self.index == 0 or self.index == 2:
                if successor.getAgentState(enemy).getPosition()[0] <= 15:
                    self.foe = enemy
            else:
                if successor.getAgentState(enemy).getPosition()[0] >= 16:
                    self.foe = enemy

        oldEnemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        oldInvaders = [a for a in oldEnemies if a.isPacman() and a.getPosition() is not None]

        if (len(invaders) > 0):
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            features['invaderDistance'] = min(dists) / 100
        for i in features:
            features[i] /= 10

        #Find out if the marked foe is a ghost
        isFoeGhost = False
        if not self.foe == -1:
            if self.index == 0 or self.index == 2:
                isFoeGhost = (successor.getAgentState(self.foe).getPosition()[0] > self.getMidpoint(gameState)[0] - 1)
            else:
                isFoeGhost = (successor.getAgentState(self.foe).getPosition()[0] < self.getMidpoint(gameState)[0])
        
        # if foe is ghost, mirror its movements to prevent it from crossing
        if not self.foe == -1 and isFoeGhost:
            yCoord = successor.getAgentState(self.foe).getPosition()[1]
            mid = self.getMidpoint(gameState)
            if self.index == 0 or self.index == 2:
                features['foeDistance'] = self.getMazeDistance(myPos, self.findNearest(gameState, (mid[0] - 1, yCoord))) / 100
            else:
                features['foeDistance'] = self.getMazeDistance(myPos, self.findNearest(gameState, (mid[0], yCoord))) / 100

        #if no foe marked, path toward center
        if self.foe == -1:
            mid = self.getMidpoint(gameState)
            if self.index == 0 or self.index == 2:
                features['foeDistance'] = self.getMazeDistance(myPos, self.findNearest(gameState, (mid[0] - 1, mid[1]))) / 100
            else:
                features['foeDistance'] = self.getMazeDistance(myPos, self.findNearest(gameState, self.getMidpoint(gameState))) / 100
        
        # feat to see if capture was made
        if (len(oldInvaders) > len(invaders)):
            features['captured'] = 1

        return features

    
    def getReward(self, state, action):
        #reward for capture
        opps = [state.getAgentState(a).getPosition() for a in self.getOpponents(state)]
        if state.generateSuccessor(self.index,action).getAgentPosition(self.index) in opps:
            return 500
        return 0

    def findNearest(self, state, coord):
        increment = 0
        if self.index == 0 or self.index == 2:
            increment = -1
        else:
            increment = 1
        walls = state.getWalls()
        x = int(coord[0])
        y = int(coord[1])
        if walls[x][y]:
            if not walls[x][y + 1]:
                return (x, y + 1)
            elif not walls[x][y - 1]:
                return (x, y - 1)
            elif not walls[x + increment][y]:
                return (x + increment, y)
            elif not walls[x + increment][y + 1]:
                return (x + increment, y + 1)
            elif not walls[x + increment][y - 1]:
                return (x + increment, y - 1)
        else:
            return (x, y)

    def getMidpoint(self, state):
        walls = state.getWalls()
        midpointX = walls.getWidth() // 2
        midpointY = walls.getHeight() // 2
        return (midpointX, midpointY)

class OffensiveQAgent(QCap):

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getFeatures(self, gameState, action):
        features = {}
        successor = self.getSuccessor(gameState, action)
        features['successorScore'] = self.getScore(successor) / 100
    
        features['bias'] = 1

        # Compute distance to the nearest food.
        foodList = self.getFood(successor).asList()
        oldFood = self.getFood(gameState).asList()

        # This should always be True, but better safe than sorry.
        if (len(foodList) > 0):
            myPos = successor.getAgentState(self.index).getPosition()
            minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
            features['distanceToFood'] = 1 / minDistance
        if(len(foodList) < len(oldFood)):
            features['ateFood'] = .8
        
        #used to prevent pathing into ghosts on offense, it doesn't work right now (I think isGhost doesn't work as intended, could be wrong though)
        opps = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        successor.getAgentState(self.index).getPosition()
        ghosts = [self.getMazeDistance(successor.getAgentState(self.index).getPosition(), opp.getPosition()) for opp in opps if opp.isGhost() and not opp.isScared()]
        if len(ghosts) == 0:
            features['ghostDistance'] = 0
            return features
        minDist = min(ghosts)
        if minDist < 4:
            features['ghostDistance'] = 1 / (minDist + 1)
        return features

    def getReward(self, state, action):
        #reward if food eaten
        if len(state.getFood().asList()) > len(state.generateSuccessor(self.index, action).getFood().asList()):
            return 500
        return 0

