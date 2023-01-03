# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""
import util
import enum


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


class SearchNode:
    def __init__(self, state, prior=None, problem=None):
        # print("SearchState init: state={}, prior={}".format(state, prior))
        self.prior = prior
        if prior is None:
            self.state = state
            self.path = []
            self.cost = 0
            self.depth = 0
            self.problem = problem
        else:
            self.state, action, step_cost = state
            self.path = prior.path.copy()
            self.path.append(action)
            self.cost = prior.cost + step_cost
            self.depth = prior.depth + 1
            self.problem = prior.problem

    # print("Created State {}".format(self))

    def __str__(self):
        return "State: {}; Path: {}; Cost: {}; Depth: {};".format(self.state, self.path, self.cost, self.depth)

    def __repr__(self):
        return "SearchNode({})".format(str(self))


def genericSearch(problem, cost_funct=lambda node: 0, depth_limit=float('inf'), find_all=False):
    solutions = []
    fringe = util.PriorityQueue()

    start = problem.getStartState()
    fringe.push(SearchNode(start, problem=problem), 0)
    # visited = set()
    # visited.add(start)
    closed = set()

    last_closed_size = 0
    last_fringe_size = 0

    while not fringe.isEmpty():
        node = fringe.pop()
        if node.state in closed:
            continue
        if len(closed) > last_closed_size + 1000:
            print("Closed: {}\nFringe: {}\nSolutions: {}".format(len(closed), len(fringe), len(solutions)))
            last_closed_size += 1000
        # print(node)
        if problem.isGoalState(node.state):
            print("Found goal state:\n{}".format(node))
            if find_all:
                solutions.append(node.path)
                continue
            else:
                return node.path
        else:
            closed.add(node.state)
        if node.depth >= depth_limit:
            continue
        for successor in problem.getSuccessors(node.state):
            state = successor[0]
            # if state not in visited:
            if state not in closed:
                new_node = SearchNode(successor, prior=node)
                fringe.push(new_node, cost_funct(new_node))
                # visited.add(state)

    if find_all:
        return solutions
    else:
        return None


def depthFirstSearch(problem, find_all=False):
    """Search the deepest nodes in the search tree first."""
    return genericSearch(problem, lambda node: -node.depth, find_all=find_all)


def breadthFirstSearch(problem, find_all=False):
    """Search the shallowest nodes in the search tree first."""
    return genericSearch(problem, lambda node: node.depth, find_all=find_all)


def iterativeDeepeningSearch(problem):
    i = 1
    while True:
        result = genericSearch(problem, lambda node: -node.depth, depth_limit=i)
        if result is not None:
            return result
        i = i + 1


def uniformCostSearch(problem, find_all=False):
    """Search the node of least total cost first."""
    return genericSearch(problem, lambda node: node.cost, find_all=find_all)
    # Alternatively and equivalently:
    # return aStarSearch(problem, nullHeuristic)


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def greedySearch(problem, heuristic=nullHeuristic, find_all=False):
    return genericSearch(problem, lambda node: heuristic(node.state, node.problem), find_all=find_all)


def aStarSearch(problem, heuristic=nullHeuristic, find_all=False):
    """Search the node that has the lowest combined cost and heuristic first."""
    return genericSearch(problem, lambda node: (node.cost + heuristic(node.state, node.problem)), find_all=find_all)


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
ids = iterativeDeepeningSearch
greedy = greedySearch
