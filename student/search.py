"""
In this file, you will implement generic search algorithms which are called by Pacman agents.
"""
from pacai.util import stack
from pacai.util import queue
from pacai.util import priorityQueue

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first [p 85].

    Your search algorithm needs to return a list of actions that reaches the goal.
    Make sure to implement a graph search algorithm [Fig. 3.7].

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    ```
    print("Start: %s" % (str(problem.startingState())))
    print("Is the start a goal?: %s" % (problem.isGoal(problem.startingState())))
    print("Start's successors: %s" % (problem.successorStates(problem.startingState())))
    ```
    """

    fringe = stack.Stack()
    # print("Start: %s" % (str(problem.startingState())))
    # print("Is the start a goal?: %s" % (problem.isGoal(problem.startingState())))
    # print("Start's successors: %s" % (problem.successorStates(problem.startingState())))
    visited = []
    if problem.isGoal(problem.startingState()):
        return visited
    # fringe.push((problem.startingState(), visited, 1))
    fringe.push((problem.startingState(), [], 1))
    while fringe.isEmpty() is False:
        node = fringe.pop()
        # direction = curr[1]
        if problem.isGoal(node[0]):
            return node[1]
        if node[0] not in visited:
            visited.append(node[0])
            successorStates = problem.successorStates(node[0])
            for state in successorStates:
                if state[0] not in visited:
                    actions = node[1] + [state[1]]
                    fringe.push((state[0], actions, state[2]))
    # raise NotImplementedError()
    return []  # failure

def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first. [p 81]
    """
    visited = []
    # pathCost = 0
    if problem.isGoal(problem.startingState()):
        return visited
    fringe = queue.Queue()
    fringe.push((problem.startingState(), [], 1))
    while fringe.isEmpty() is False:
        node = fringe.pop()
        if problem.isGoal(node[0]):
            return node[1]
        if node[0] not in visited:
            visited.append(node[0])
            successorStates = problem.successorStates(node[0])
            for state in successorStates:
                if state[0] not in visited:
                    actions = node[1] + [state[1]]
                    fringe.push((state[0], actions, state[2]))
    return []  # failure

def uniformCostSearch(problem):
    """
    Search the node of least total cost first.
    """
    fringe = priorityQueue.PriorityQueue()
    visited = []
    if problem.isGoal(problem.startingState()):
        return visited
    fringe.push((problem.startingState(), [], 1), 0)
    while fringe.isEmpty() is False:
        node = fringe.pop()
        if problem.isGoal(node[0]):
            return node[1]
        if node[0] not in visited:
            visited.append(node[0])
            successorStates = problem.successorStates(node[0])
            for state in successorStates:
                if state[0] not in visited:
                    actions = node[1] + [state[1]]
                    cost = node[2] + state[2]
                    fringe.push((state[0], actions, cost), cost)
    # *** Your Code Here ***
    # raise NotImplementedError()
    return []  # failure

def aStarSearch(problem, heuristic):
    """
    Search the node that has the lowest combined cost and heuristic first.
    """
    fringe = priorityQueue.PriorityQueue()
    visited = []
    if problem.isGoal(problem.startingState()):
        return visited
    fringe.push((problem.startingState(), [], 1), 0)
    while fringe.isEmpty() is False:
        node = fringe.pop()
        if problem.isGoal(node[0]):
            return node[1]
        if node[0] not in visited:
            visited.append(node[0])
            successorStates = problem.successorStates(node[0])
            for state in successorStates:
                if state[0] not in visited:
                    actions = node[1] + [state[1]]
                    cost = node[2] + state[2]
                    hCost = cost + heuristic(state[0], problem)
                    fringe.push((state[0], actions, cost), hCost)
    # *** Your Code Here ***
    # raise NotImplementedError()
    return []  # failure
