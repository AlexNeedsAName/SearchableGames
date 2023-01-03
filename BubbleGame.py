import math

import cv2
import numpy as np

import search

key = [str(i)[0] for i in range(0, 10)] \
      + [chr(i) for i in range(ord('A'), ord('Z') + 1)] \
      + [chr(i) for i in range(ord('a'), ord('z') + 1)]


def getTopBall(tube):
    for ball in tube[::-1]:
        if ball is not None:
            return ball
    return None


def euclid(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def taxicab(a, b):
    return sum(math.fabs(x - y) for x, y in zip(a, b))


def update_average(sample, average, size):
    return tuple((s + a * size) / (size + 1) for s, a in zip(sample, average))


# Creates a pallet of colors from samples of colors, merging samples that are close enough
class Pallet:
    def __init__(self, dist_function=math.dist, min_dist=30):
        self._colors = []
        self._samples = []
        self._dist = dist_function
        self._min_dist = min_dist

    # Get the index of a sample color. If the color does not match any existing colors, it will be added to the pallet
    def index(self, sample):
        for i, color in enumerate(self._colors):
            if self._dist(sample, color) < self._min_dist:
                self._colors[i] = update_average(sample, color, self._samples[i])
                self._samples[i] += 1
                return i

        self._colors.append(sample)
        self._samples.append(1)
        return len(self._colors) - 1

    def add(self, sample):
        self.index(sample)
        return self

    def __len__(self):
        return len(self._colors)

    def __getitem__(self, index):
        return tuple(int(x) for x in self._colors[index])

    def __contains__(self, sample):
        for color in self._colors:
            if self._dist(sample, color) < self._min_dist:
                return True
        return False

    def __str__(self):
        return str(self._colors)


empty = Pallet().add((255, 255, 255)).add((232, 232, 232))
print("Empty: {}".format(empty))


class Size:
    def __init__(self, tubes, zero, radius=36.0, rows=1, cols=None, dx=0, dy=0):
        self.tubes = tubes
        self.zero = zero
        self.radius = radius
        self.rows = rows
        if cols is None:
            cols = math.ceil(tubes / rows)
        self.cols = cols
        self.dx = dx
        self.dy = dy


window_size = (750, 1334)
sizes = {
    5: Size(tubes=5, zero=(96, 796), radius=36, rows=1, dx=141, dy=0),
    6: Size(tubes=6, zero=(129, 544), radius=36, rows=2, dx=246, dy=490),
    7: Size(tubes=7, zero=(94, 545), radius=36, rows=2, dx=190, dy=490),
    8: Size(tubes=8, zero=(94, 545), radius=36, rows=2, dx=190, dy=490),
    9: Size(tubes=9, zero=(80, 545), radius=36, rows=2, dx=147, dy=490),
    10: Size(tubes=10, zero=(80, 545), radius=36, rows=2, dx=147, dy=490),
    11: Size(tubes=11, zero=(58, 564), radius=28.5, rows=2, dx=126, dy=418),
    12: Size(tubes=12, zero=(58, 564), radius=28.5, rows=2, dx=126, dy=418),
    14: Size(tubes=14, zero=(55, 577), radius=24.5, rows=2, dx=108, dy=428),
    15: Size(tubes=15, zero=(91, 437), radius=24.5, rows=3, dx=143, dy=359),
}


class BubbleSortSearch(search.SearchProblem):
    def __init__(self, start_state):
        print("Created search problem for {} {}".format(len(start_state.state), start_state))
        self.start_state = start_state

    def getStartState(self):
        return self.start_state

    def isGoalState(self, state):
        return state.isWinState()

    def getSuccessors(self, state):
        return state.generateSuccessors()

    def getCostOfActions(self, actions):
        return len(actions)


class BubbleSortGame:
    def __init__(self, tubes, source, colors=None, positions=None, height=4, moves=0):
        self.state = []
        self.moves = moves

        if colors is None:
            colors = Pallet()
        self.colors = colors
        self.height = height

        if positions is None:
            size = sizes[tubes]
            positions = []
            for i in range(tubes):
                tube = []
                row, col = divmod(i, size.cols)
                last_row = row == size.rows - 1 and size.rows * size.cols != tubes
                for j in range(height + 1):
                    x, y = size.zero
                    if not last_row:
                        x += col * size.dx
                    else:
                        x += col * size.dx + int(size.dx / 2)
                    y += row * size.dy - 2 * j * size.radius
                    tube.append((int(x), int(y)))
                positions.append(tube)
        self.positions = positions

        if isinstance(source, list):
            assert len(source) == tubes
            self.state = source
        else:
            img = cv2.imread(source, cv2.IMREAD_COLOR)
            for i in range(tubes):
                tube = []
                for j in range(height):
                    x, y = self.positions[i][j]
                    color = img[y, x]
                    # print("Color = {} = {}".format(img[y, x], color))
                    if color in empty:
                        break
                    color_index = self.colors.index(color)
                    tube.append(color_index)
                self.state.append(tube)
            print(len(self.colors), len(self.state))
            print(colors)
            cv2.imshow("screenshot", img)
            self.show("Steps")
            cv2.waitKey()
            assert len(self.colors) == len(self.state) - 2

        self.win = ' '.join([key[color] * 4 for color in range(1, len(self.state) - 1)] + ["____"] * 2)

    def generateSuccessors(self):
        successors = []
        top_colors = [getTopBall(tube) for tube in self.state]
        for i, source in enumerate(self.state):
            if len(source) == 0:
                continue
            for j, dest in enumerate(self.state):
                if j == i:
                    continue
                if len(dest) == 0 or (len(dest) < self.height and top_colors[i] == top_colors[j]):
                    successors.append((self.doMove(i, j), (i, j), 1))
        return successors

    def doMove(self, tube_from, tube_to):
        new_state = [tube.copy() for tube in self.state]
        new_state[tube_to].append(new_state[tube_from].pop())
        return BubbleSortGame(len(new_state), new_state, self.colors, self.positions, self.height, self.moves + 1)

    def isWinState(self):
        # return hash(self) == hash(self.win)

        for tube in self.state:
            # Each tube must either be empty or full
            if len(tube) == 0:
                continue
            if len(tube) != self.height:
                return False

            # Full tubes should only contain one color
            bottom_ball = tube[0]
            for ball in tube[1:]:
                if ball != bottom_ball:
                    return False
        return True

    def show(self, window):
        print("Showing {} on window {}".format(self, window))
        width, height = window_size
        img = np.ones((height, width, 3), dtype=np.uint8) * 255
        # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for i, tube in enumerate(self.state):
            for j, color_key in enumerate(tube):
                x, y = self.positions[i][j]
                color = self.colors[color_key]
                # print("Color = {} = {} = {}".format(color_key, self.colors[color_key], color))
                cv2.circle(img, center=(x, y), radius=int(sizes[len(self.state)].radius - 2),
                           color=color, thickness=-1)
                cv2.circle(img, center=(x, y), radius=int(sizes[len(self.state)].radius - 2),
                           color=(0, 0, 0), thickness=1)
                cv2.putText(img, str(key[color_key]), (x - 9, y + 9), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
        cv2.imshow(window, img)

    def __str__(self):
        tube_strings = [''.join([key[color] for color in tube] + ['_' for _ in range(self.height - len(tube))])
                        for tube in self.state]
        tube_strings.sort()
        return ' '.join(tube_strings)

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return hash(self) == hash(other)


def minimize_color_changes(state, problem=None):
    approx = 0
    for tube in state.state:
        if len(tube) == 0:
            continue
        last = tube[0]
        for ball in tube[1:]:
            if last != ball:
                approx += 1
                last = ball
    return approx


def minimize_color_changes2(state, problem=None):
    approx = 0
    for tube in state.state:
        if len(tube) == 0:
            continue
        last = tube[0]
        for ball in tube[1:]:
            if last != ball:
                approx *= 2
                last = ball
    return approx


def run_test(num_tubes, name=None, wait=False):
    if name is None:
        name = num_tubes
    start_state = BubbleSortGame(num_tubes, "ref/{}.PNG".format(name))
    problem = BubbleSortSearch(start_state)
    # solution = search.astar(problem)
    # solutions = search.astar(problem, find_all=True)
    solutions = search.astar(problem, heuristic=minimize_color_changes, find_all=True)
    if len(solutions) == 0:
        print("No solution")
    else:
        print("Found {} solutions".format(len(solutions)))
    # for solution in solutions:
    #     print("{} Moves".format(len(solution)))
    for solution in solutions[:1]:
        state = BubbleSortGame(len(start_state.state), start_state.state, start_state.colors, start_state.positions,
                               start_state.height)
        for tube_from, tube_to in solution:
            state = state.doMove(tube_from, tube_to)
            state.show("Steps")
            if wait:
                cv2.waitKey()
            else:
                cv2.waitKey(100)
        print(len(solution), "steps")
    cv2.waitKey()


if __name__ == "__main__":
    cv2.namedWindow("Steps")
    cv2.namedWindow("screenshot")
    cv2.setWindowProperty("screenshot", cv2.WND_PROP_TOPMOST, 1)
    cv2.setWindowProperty("Steps", cv2.WND_PROP_TOPMOST, 1)
    cv2.setWindowProperty("screenshot", cv2.WND_PROP_TOPMOST, 0)
    cv2.setWindowProperty("Steps", cv2.WND_PROP_TOPMOST, 0)
    cv2.moveWindow("Steps", 425, 0)
    cv2.moveWindow("screenshot", 425 * 2, 0)
    # run_test(5, wait=True)
    # run_test(6)
    # run_test(7)
    # run_test(8)
    # run_test(11, wait=True)
    # run_test(12)
    # run_test(14)
    # run_test(15)
    run_test(15, "5-3")
    # print(solution)
    # cv2.waitKey()
