from pacai.agents.learning.value import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
    A value iteration agent.

    Make sure to read `pacai.agents.learning` before working on this class.

    A `ValueIterationAgent` takes a `pacai.core.mdp.MarkovDecisionProcess` on initialization,
    and runs value iteration for a given number of iterations using the supplied discount factor.

    Some useful mdp methods you will use:
    `pacai.core.mdp.MarkovDecisionProcess.getStates`,
    `pacai.core.mdp.MarkovDecisionProcess.getPossibleActions`,
    `pacai.core.mdp.MarkovDecisionProcess.getTransitionStatesAndProbs`,
    `pacai.core.mdp.MarkovDecisionProcess.getReward`.

    Additional methods to implement:

    `pacai.agents.learning.value.ValueEstimationAgent.getQValue`:
    The q-value of the state action pair (after the indicated number of value iteration passes).
    Note that value iteration does not necessarily create this quantity,
    and you may have to derive it on the fly.

    `pacai.agents.learning.value.ValueEstimationAgent.getPolicy`:
    The policy is the best action in the given state
    according to the values computed by value iteration.
    You may break ties any way you see fit.
    Note that if there are no legal actions, which is the case at the terminal state,
    you should return None.
    """

    def __init__(self, index, mdp, discountRate = 0.9, iters = 100, **kwargs):
        super().__init__(index, **kwargs)
        self.mdp = mdp
        self.discountRate = discountRate
        self.iters = iters
        self.values = {}
        # A dictionary which holds the q-values for each state.
        # Compute the values here.
        for s in self.mdp.getStates():
            self.values[s] = 0
        for iter in range(0, iters):
            currValues = {}
            for s in self.mdp.getStates():
                currValues[s] = 0
            # print("s: ", s)
            for state in self.mdp.getStates():
                # bV = -10000
                # if self.mdp.isTerminal(state):
                #     continue
                actions = self.mdp.getPossibleActions(state)
                # print("actions: ", actions, state)
                values = []
                for action in actions:
                    values.append(self.getQValue(state, action))
                    # value = self.getQValue(state, action)
                    # currValues[state] = bV
                # print("values: ", values)
                if len(values) == 0:
                    bV = 0
                else:
                    bV = max(values)
                currValues[state] = bV
                # print("bV: ", bV)
            for state in mdp.getStates():
                self.values[state] = currValues[state]
            # self.values = currValues

    def getQValue(self, state, action):
        """
        The q-value of the state action pair (after the indicated number of value iteration passes).
        Note that value iteration does not necessarily create this quantity,
        and you may have to derive it on the fly.
        """
        qV = 0
        for stateAndProb in self.mdp.getTransitionStatesAndProbs(state, action):
            currState = stateAndProb[0]
            currProb = stateAndProb[1]
            reward = self.mdp.getReward(state, action, currState)
            # print("currProb: ", currProb)
            # print("currState: ", self.values[currState[0]])
            qV += currProb * (reward + self.discountRate * self.getValue(currState))
        return qV

    def getPolicy(self, state):
        """
        The policy is the best action in the given state
        according to the values computed by value iteration.
        You may break ties any way you see fit.
        Note that if there are no legal actions, which is the case at the terminal state,
        you should return None.
        """
        if self.mdp.isTerminal(state):
            return None
        valuePairs = []
        actions = self.mdp.getPossibleActions(state)
        for action in actions:
            value = self.getQValue(state, action)
            valuePairs.append((value, action))
        bestPair = sorted(valuePairs, key=lambda x: x[0], reverse=True)[0]
        bestAction = bestPair[1]
        return bestAction

    def getValue(self, state):
        """
        Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def getAction(self, state):
        """
        Returns the policy at the state (no exploration).
        """
        return self.getPolicy(state)
