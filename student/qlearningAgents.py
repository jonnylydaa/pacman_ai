from pacai.agents.learning.reinforcement import ReinforcementAgent
from pacai.util import reflection
from pacai.util import probability
import random

class QLearningAgent(ReinforcementAgent):
    """
    A Q-Learning agent.

    Some functions that may be useful:

    `pacai.agents.learning.reinforcement.ReinforcementAgent.getAlpha`:
    Get the learning rate.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.getDiscountRate`:
    Get the discount rate.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.getEpsilon`:
    Get the exploration probability.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.getLegalActions`:
    Get the legal actions for a reinforcement agent.

    `pacai.util.probability.flipCoin`:
    Flip a coin (get a binary value) with some probability.

    `random.choice`:
    Pick randomly from a list.

    Additional methods to implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Compute the action to take in the current state.
    With probability `pacai.agents.learning.reinforcement.ReinforcementAgent.getEpsilon`,
    we should take a random action and take the best policy action otherwise.
    Note that if there are no legal actions, which is the case at the terminal state,
    you should choose None as the action.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.update`:
    The parent class calls this to observe a state transition and reward.
    You should do your Q-Value update here.
    Note that you should never call this function, it will be called on your behalf.

    DESCRIPTION: This QLearningAgent learns best moves based on trial runs. It updates
    q values using the update function, as it learns the grid.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)
        # You can initialize Q-values here.
        self.values = {}

    def getQValue(self, state, action):
        """
        Get the Q-Value for a `pacai.core.gamestate.AbstractGameState`
        and `pacai.core.directions.Directions`.
        Should return 0.0 if the (state, action) pair has never been seen.
        """
        pair = (state, action)
        if pair in self.values:
            return self.values[pair]
        return 0.0

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
        values = []
        for action in self.getLegalActions(state):
            values.append(self.getQValue(state, action))
        if len(values) > 0:
            return max(values)
        else:
            return 0.0

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
        actions = []
        bV = self.getValue(state)
        for action in self.getLegalActions(state):
            if self.getQValue(state, action) == bV:
                actions.append(action)
        if len(actions) > 0:
            return random.choice(actions)
        else:
            # no legal actions
            return None

    def getAction(self, state):
        """
        Compute the action to take in the current state.
        With probability `pacai.agents.learning.reinforcement.ReinforcementAgent.getEpsilon`,
        we should take a random action and take the best policy action otherwise.
        Note that if there are no legal actions, which is the case at the terminal state,
        you should choose None as the action.
        """
        # print("ep: ", self.getEpsilon())
        if probability.flipCoin(self.getEpsilon()) is True:
            action = random.choice(self.getLegalActions(state))
            return action
        else:
            action = self.getPolicy(state)
            # getPolicy takes care of terminal state possibility
            return action

    def update(self, state, action, nextState, reward):
        """
        The parent class calls this to observe a state transition and reward.
        You should do your Q-Value update here.
        Note that you should never call this function, it will be called on your behalf.
        """
        pair = (state, action)
        # current = self.values[pair]
        next = self.getValue(nextState)
        q = self.getQValue(state, action)
        new = (1 - self.getAlpha()) * q + self.getAlpha() * (reward + self.getDiscountRate() * next)
        self.values[pair] = new
        # return self.values

class PacmanQAgent(QLearningAgent):
    """
    Exactly the same as `QLearningAgent`, but with different default parameters.
    """

    def __init__(self, index, epsilon = 0.05, gamma = 0.8, alpha = 0.2, numTraining = 0, **kwargs):
        kwargs['epsilon'] = epsilon
        kwargs['gamma'] = gamma
        kwargs['alpha'] = alpha
        kwargs['numTraining'] = numTraining

        super().__init__(index, **kwargs)

    def getAction(self, state):
        """
        Simply calls the super getAction method and then informs the parent of an action for Pacman.
        Do not change or remove this method.
        """
        action = super().getAction(state)
        self.doAction(state, action)
        return action

class ApproximateQAgent(PacmanQAgent):
    """
    An approximate Q-learning agent.

    You should only have to overwrite `QLearningAgent.getQValue`
    and `pacai.agents.learning.reinforcement.ReinforcementAgent.update`.
    All other `QLearningAgent` functions should work as is.

    Additional methods to implement:

    `QLearningAgent.getQValue`:
    Should return `Q(state, action) = w * featureVector`,
    where `*` is the dotProduct operator.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.update`:
    Should update your weights based on transition.

    DESCRIPTION: I implement a weight value for each feature in a state
    action pair feature vector, which approximates weights. This helps
    the agent learn.
    """

    def __init__(self, index, extractor='pacai.core.featureExtractors.IdentityExtractor', **kwargs):
        super().__init__(index, **kwargs)
        self.featExtractor = reflection.qualifiedImport(extractor)
        # You might want to initialize weights here.
        self.weights = {}
        # print("feat extract: ", self.featExtractor())
        # for s in self.featExtractor():
        #     self.weights[s] = 0

    def getQValue(self, state, action):
        """
        Should return `Q(state, action) = w * featureVector`,
        where `*` is the dotProduct operator.
        """
        feats = self.featExtractor.getFeatures(self, state, action)
        # self.weights[(state, action)] = 0
        # print("feats: ", feats)
        q = 0
        for feat in feats:
            if feat not in self.weights:
                self.weights[feat] = 0.0
            # print("feat: ", feats[feat])
            # print("self weights: ", self.weights[feat])
            q += self.weights[feat] * feats[feat]
        return q

    def update(self, state, action, nextState, reward):
        """
        Should update your weights based on transition.
        """
        correct = (reward + self.getDiscountRate() * self.getValue(nextState))
        correct = correct - self.getQValue(state, action)
        feats = self.featExtractor.getFeatures(self, state, action)
        for feat in feats:
            self.weights[feat] = self.weights[feat] + self.getAlpha() * correct * feats[feat]
        # return self.weights

    def final(self, state):
        """
        Called at the end of each game.
        """
        # Call the super-class final method.
        super().final(state)
        # Did we finish training?
        # if self.episodesSoFar == self.numTraining:
        # You might want to print your weights here for debugging.
        # *** Your Code Here ***
        # print("weights: ", self.weights)
        # return
        # return
