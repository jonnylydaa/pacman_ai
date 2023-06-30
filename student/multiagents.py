import random

from pacai.agents.base import BaseAgent
from pacai.agents.search.multiagent import MultiAgentSearchAgent
from pacai.core import distance


class ReflexAgent(BaseAgent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.
    You are welcome to change it in any way you see fit,
    so long as you don't touch the method headers.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        `ReflexAgent.getAction` chooses among the best options according to the evaluation function.

        Just like in the previous project, this method takes a
        `pacai.core.gamestate.AbstractGameState` and returns some value from
        `pacai.core.directions.Directions`.
        """

        # Collect legal moves.
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions.
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best.

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current `pacai.bin.pacman.PacmanGameState`
        and an action, and returns a number, where higher numbers are better.
        Make sure to understand the range of different values before you combine them
        in your evaluation function.
        """
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        # Useful information you can extract.
        newPosition = successorGameState.getPacmanPosition()
        # oldFood = currentGameState.getFood()
        # newGhostStates = successorGameState.getGhostStates()
        ghostPositions = successorGameState.getGhostPositions()
        # newScaredTimes = [ghostState.getScaredTimer() for ghostState in newGhostStates]
        score = successorGameState.getScore()
        # *** Your Code Here ***
        foods = successorGameState.getFood()
        foodList = foods.asList()
        minimum = 1000000
        # find closest food
        for food in foodList:
            currDistance = distance.manhattan(food, newPosition)
            if currDistance < minimum:
                minimum = currDistance
        # make sure ghost isn't close
        for gPosition in ghostPositions:
            ghostDist = distance.manhattan(newPosition, gPosition)
            if ghostDist < 2:
                return -1000000
        recip = 1 / minimum
        return score + recip


class MinimaxAgent(MultiAgentSearchAgent):
    """
    A minimax agent.

    Here are some method calls that might be useful when implementing minimax.

    `pacai.core.gamestate.AbstractGameState.getNumAgents()`:
    Get the total number of agents in the game

    `pacai.core.gamestate.AbstractGameState.getLegalActions`:
    Returns a list of legal actions for an agent.
    Pacman is always at index 0, and ghosts are >= 1.

    `pacai.core.gamestate.AbstractGameState.generateSuccessor`:
    Get the successor game state after an agent takes an action.

    `pacai.core.directions.Directions.STOP`:
    The stop direction, which is always legal, but you may not want to include in your search.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the minimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)
        self.index = index
        # self.depth = 2
        self.evalFunction = self.getEvaluationFunction()
        self.tD = self.getTreeDepth()

    def minValue(self, currState, agentIndex, depth, agentAmount):
        # for min node
        v = 10000000
        if depth == self.tD:
            return self.evalFunction(currState)
        if currState.isOver():
            return self.evalFunction(currState)
        if agentIndex == (agentAmount - 1):
            currActions = currState.getLegalActions(agentIndex)
            # actions.remove('Stop')
            for action in currActions:
                next = currState.generateSuccessor(agentIndex, action)
                v = min(v, self.maxValue(next, 0, depth + 1, agentAmount))
            return v
        else:
            currActions = currState.getLegalActions(agentIndex)
            for action in currActions:
                next = currState.generateSuccessor(agentIndex, action)
                v = min(v, self.minValue(next, agentIndex + 1, depth, agentAmount))
            return v

    def maxValue(self, currState, agentIndex, depth, agentAmount):
        # for max node
        v = -10000000
        if depth == self.tD:
            return self.evalFunction(currState)
        if currState.isOver():
            return self.evalFunction(currState)
        if agentIndex == 0:
            currActions = currState.getLegalActions(agentIndex)
            currActions.remove('Stop')
            for action in currActions:
                next = currState.generateSuccessor(agentIndex, action)
                v = max(v, self.minValue(next, agentIndex + 1, depth, agentAmount))
            return v

    def getAction(self, state):
        # depth = MultiAgentSearchAgent.getTreeDepth
        # evalfn = self.getEvaluationFunction()
        agentAmount = state.getNumAgents()
        # tree_depth = self.getTreeDepth()
        # print("index: ", self.index)
        actions = []
        currActions = state.getLegalActions()
        currActions.remove('Stop')
        for action in currActions:
            next = state.generateSuccessor(self.index, action)
            agentID = next.getLastAgentMoved()
            currentAct = (action, self.minValue(next, agentID + 1, 0, agentAmount))
            actions.append(currentAct)
        # print("actions: ", actions)
        # takes maximum value and returns associated action (best action)
        actions = sorted(actions, key=lambda x: x[1])
        # print("actions sorted: ", actions)
        # bestAction = max(actions)
        bestAction = actions[-1]
        # print("atT: ", bestAction)
        bestAction = bestAction[0]
        return bestAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    A minimax agent with alpha-beta pruning.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the minimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)
        self.index = index
        # self.depth = 2
        self.evalFunction = self.getEvaluationFunction()
        self.tD = self.getTreeDepth()

    def minValue(self, currState, agentIndex, depth, agentAmount, a, b):
        # for min node
        v = 10000000
        if depth == self.tD:
            return self.evalFunction(currState)
        if currState.isOver():
            return self.evalFunction(currState)
        if agentIndex == (agentAmount - 1):
            currActions = currState.getLegalActions(agentIndex)
            # actions.remove('Stop')
            for action in currActions:
                next = currState.generateSuccessor(agentIndex, action)
                v = min(v, self.maxValue(next, 0, depth + 1, agentAmount, a, b))
                if v <= a:
                    return v
                b = min(b, v)
            return v
        else:
            currActions = currState.getLegalActions(agentIndex)
            for action in currActions:
                next = currState.generateSuccessor(agentIndex, action)
                v = min(v, self.minValue(next, agentIndex + 1, depth, agentAmount, a, b))
                # checks pruning
                if v <= a:
                    return v
                b = min(b, v)
            return v

    def maxValue(self, currState, agentIndex, depth, agentAmount, a, b):
        # for max node
        v = -10000000
        if depth == self.tD:
            return self.evalFunction(currState)
        if currState.isOver():
            return self.evalFunction(currState)
        if agentIndex == 0:
            currActions = currState.getLegalActions(agentIndex)
            currActions.remove('Stop')
            for action in currActions:
                next = currState.generateSuccessor(agentIndex, action)
                v = max(v, self.minValue(next, agentIndex + 1, depth, agentAmount, a, b))
                # checks pruning
                if v >= b:
                    return v
                a = max(a, v)
            return v

    def getAction(self, state):
        # depth = MultiAgentSearchAgent.getTreeDepth
        # evalfn = self.getEvaluationFunction()
        agentAmount = state.getNumAgents()
        # tree_depth = self.getTreeDepth()
        # print("index: ", self.index)
        actions = []
        currActions = state.getLegalActions()
        a = -10000000
        b = 10000000
        currActions.remove('Stop')
        for action in currActions:
            next = state.generateSuccessor(self.index, action)
            agentID = next.getLastAgentMoved()
            currentAct = (action, self.minValue(next, agentID + 1, 0, agentAmount, a, b))
            actions.append(currentAct)
        # print("actions: ", actions)
        # takes maximum value and returns associated action (best action)
        actions = sorted(actions, key=lambda x: x[1])
        # print("actions sorted: ", actions)
        bestAction = actions[-1]
        # print("atT: ", bestAction)
        bestAction = bestAction[0]
        return bestAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    An expectimax agent.

    All ghosts should be modeled as choosing uniformly at random from their legal moves.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the expectimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)
        self.index = index
        # self.depth = 2
        self.evalFunction = self.getEvaluationFunction()
        self.tD = self.getTreeDepth()

    def maxValue(self, currState, agentIndex, depth, agentAmount):
        # for max node
        if depth == self.tD:
            return self.evalFunction(currState)
        if currState.isOver():
            return self.evalFunction(currState)
        v = 0
        if agentIndex == 0:
            v = -10000000
            currActions = currState.getLegalActions(agentIndex)
            currActions.remove('Stop')
            for action in currActions:
                next = currState.generateSuccessor(agentIndex, action)
                v = max(v, self.value(next, agentIndex + 1, depth, agentAmount, True))
        return v

    def expValue(self, currState, agentIndex, depth, agentAmount):
        # for probability node
        if depth == self.tD:
            return self.evalFunction(currState)
        if currState.isOver():
            return self.evalFunction(currState)
        v = 0
        # checks if agent is pacman or ghost
        if agentIndex != 0:
            actions = currState.getLegalActions(agentIndex)
            for action in actions:
                next = currState.generateSuccessor(agentIndex, action)
                v += self.value(next, 0, depth + 1, agentAmount, False)
        elif agentIndex % agentAmount != 0:
            actions = currState.getLegalActions(agentIndex)
            for action in actions:
                next = currState.generateSuccessor(agentIndex, action)
                v += self.value(next, agentIndex + 1, depth, agentAmount, True)
        # takes probability
        prob = v / len(actions)
        return prob

    def value(self, currState, agentIndex, depth, agentAmount, probNode):
        if depth == self.tD:
            return self.evalFunction(currState)
        if currState.isOver():
            return self.evalFunction(currState)
        if (probNode is not True) and (agentIndex == 0):
            return self.maxValue(currState, agentIndex, depth, agentAmount)
        else:
            # probability node
            return self.expValue(currState, agentIndex, depth, agentAmount)

    def getAction(self, state):
        # depth = MultiAgentSearchAgent.getTreeDepth
        # evalfn = self.getEvaluationFunction()
        agentAmount = state.getNumAgents()
        # tree_depth = self.getTreeDepth()
        # print("index: ", self.index)
        actions = []
        currActions = state.getLegalActions()
        currActions.remove('Stop')
        for action in currActions:
            next = state.generateSuccessor(self.index, action)
            agentID = next.getLastAgentMoved()
            currentAct = (action, self.value(next, agentID + 1, 0, agentAmount, True))
            actions.append(currentAct)
        # print("actions: ", actions)
        # takes maximum value and returns associated action (best action)
        actions = sorted(actions, key=lambda x: x[1])
        # print("actions sorted: ", actions)
        bestAction = actions[-1]
        # print("atT: ", bestAction)
        bestAction = bestAction[0]
        return bestAction

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable evaluation function.

    DESCRIPTION: <write something here so we know what you did>
    In my betterEvaluationFunction, I look at the distance to the closest food, the amount
    of food left in the game, the distance of the closest ghost, if the game is a win
    or loss, and the amount of capsules left in the game. With these attributes, I weigh
    them with different multipliers, multiplying these values with the reciprocal value
    of each one of these attributes. I then add all of these values together to get the
    final evaluation function score. The amount of food left in the game and the current
    game state of loss or win are weighted the heaviest in this evaluation function.
    """
    position = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    foodAmount = currentGameState.getNumFood()
    foodList = food.asList()
    gameScore = currentGameState.getScore()
    caps = currentGameState.getCapsules()
    capsAmount = len(caps)
    # print(capsAmount)
    # print("game score: ", gameScore)
    # successorGameState = currentGameState.generatePacmanSuccessor(action)
    # Useful information you can extract.
    # newPosition = successorGameState.getPacmanPosition()
    # oldFood = currentGameState.getFood()
    # newGhostStates = successorGameState.getGhostStates()
    ghostPositions = currentGameState.getGhostPositions()
    # newScaredTimes = [ghostState.getScaredTimer() for ghostState in newGhostStates]
    # score = successorGameState.getScore()
    # *** Your Code Here ***
    minimum = 1000000
    # finds closest food item distance
    for food in foodList:
        currDistance = distance.manhattan(food, position)
        if currDistance < minimum:
            minimum = currDistance
    # finds closest ghost distance
    for gPosition in ghostPositions:
        ghostDist = distance.manhattan(position, gPosition)
        if ghostDist < 2:
            return -1000000
    # reciprocal values to be incorporated into final evaluation score
    minRecip = 1 / (minimum + 1)
    foodRecip = 1 / (foodAmount + 1)
    ghostRecip = 1 / (ghostDist + 1)
    capsRecip = 1 / (capsAmount + 1)
    # print("ghost: ", ghostDist)
    gameScenario = 0   # either win or lose
    # checks game status
    if currentGameState.isWin():
        gameScenario += 35000
    elif currentGameState.isLose():
        gameScenario -= 35000
    # print("game scenario: ", gameScenario)
    # print("min: ", minimum)
    # arbitrary multipliers based on subjective importance of attributes
    score = (minRecip * 800) + (foodRecip * 80000) + (capsRecip * 2100) + (ghostRecip * -10)
    return score + gameScenario + gameScore

class ContestAgent(MultiAgentSearchAgent):
    """
    Your agent for the mini-contest.

    You can use any method you want and search to any depth you want.
    Just remember that the mini-contest is timed, so you have to trade off speed and computation.

    Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
    just make a beeline straight towards Pacman (or away if they're scared!)

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)
