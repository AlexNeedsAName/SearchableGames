import copy
import itertools
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

window_size = (750, 1334)


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


win_hashes = {}


def win_hash(tubes):
    try:
        return win_hashes[tubes]
    except KeyError:
        win_hashes[tubes] = hash(' '.join([key[color] * 4 for color in range(0, tubes - 2)] + ["____"] * 2))
        return win_hashes[tubes]


# From image
# From parent

class BubbleSortGame:
    def __init__(self, source, height=4):
        self.state = []
        self.colors = Pallet()
        self.height = height

        img = cv2.imread(source, cv2.IMREAD_COLOR)

        # Find the positions of each slot
        self.positions = []
        img = cv2.imread(source, cv2.IMREAD_COLOR)
        thresh = cv2.inRange(img, (238 - 10, 192 - 10, 87 - 10), (238 + 10, 192 + 10, 87 + 10))  # TODO: hardcoded color
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if not (555 < area < 1155):  # TODO: hardcoded areas
                continue
            x, y, w, h = cv2.boundingRect(contour)
            # center = (int(x + w / 2), int(y + h / 2))
            # cv2.circle(img, center=center, radius=5, color=(0, 0, 0), thickness=-1)
            # TODO: Hardcoded proportions
            self.radius = int(0.34 * w)
            spacing = 0.70 * w
            offset = 2.76 * w

            x = int(x + w / 2)
            y = y + h / 2 + offset
            self.positions.append([(x, int(y - i * spacing)) for i in itertools.chain(range(height), (height + 1.5,))])

        # Record the colors in each spot
        for i in range(len(self.positions)):
            tube = []
            for j in range(height):
                x, y = self.positions[i][j]
                color = img[y, x]
                if color in empty:
                    break
                color_index = self.colors.index(color)
                tube.append(color_index)
            self.state.append(tube)
        cv2.imshow("screenshot", img)
        assert len(self.colors) == len(self.state) - 2

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
        result = copy.copy(self)
        new_state = [tube.copy() for tube in self.state]
        new_state[tube_to].append(new_state[tube_from].pop())
        result.state = new_state
        return result

    def removeFrom(self, tube_from):
        result = copy.copy(self)
        new_state = [tube.copy() for tube in self.state]
        new_state[tube_from].pop()
        result.state = new_state
        return result

    def top(self, tube):
        return getTopBall(self.state[tube])

    def count(self, tube):
        return len(self.state[tube])

    def isWinState(self):
        return hash(self) == win_hash(len(self.state))

        # for tube in self.state:
        #     # Each tube must either be empty or full
        #     if len(tube) == 0:
        #         continue
        #     if len(tube) != self.height:
        #         return False
        #
        #     # Full tubes should only contain one color
        #     bottom_ball = tube[0]
        #     for ball in tube[1:]:
        #         if ball != bottom_ball:
        #             return False
        # return True

    def show(self):
        width, height = window_size
        img = np.ones((height, width, 3), dtype=np.uint8) * 255
        for i, tube in enumerate(self.state):
            for j, color_key in enumerate(tube):
                x, y = self.positions[i][j]
                color = self.colors[color_key]
                cv2.circle(img, center=(x, y), radius=self.radius, color=color, thickness=-1)
                cv2.circle(img, center=(x, y), radius=self.radius, color=(0, 0, 0), thickness=1)
                cv2.putText(img, str(key[color_key]), (x - 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
        return img

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


def bezier_ease(t):
    return t * t * (3 - 2 * t)


def weighted_average(t1, t2, w):
    return tuple(x * (1 - w) + y * w for x, y in zip(t1, t2))


def run_test(num_tubes, name=None, wait=False):
    if name is None:
        name = num_tubes
    start_state = BubbleSortGame("ref/{}.PNG".format(name))
    cv2.imshow("Steps", start_state.show())
    cv2.waitKey()
    problem = BubbleSortSearch(start_state)
    solutions = search.astar(problem, heuristic=minimize_color_changes, find_all=True)
    if len(solutions) == 0:
        print("No solution")
    else:
        print("Found {} solutions".format(len(solutions)))
    # for solution in solutions:
    #     print("{} Moves".format(len(solution)))
    for solution in solutions[:1]:
        state = start_state
        for tube_from, tube_to in solution:
            keyframes = (
                state.positions[tube_from][state.count(tube_from) - 1], state.positions[tube_from][state.height],
                state.positions[tube_to][state.height], state.positions[tube_to][state.count(tube_to)]
            )
            radius = state.radius
            color_key = state.top(tube_from)
            color = state.colors[color_key]
            color_key = str(key[color_key])
            base_img = state.removeFrom(tube_from).show()
            state = state.doMove(tube_from, tube_to)
            if wait:
                cv2.waitKey()

            frames = (4, 12, 4)
            framerate = 60
            for p0, p1, frames in zip(keyframes, keyframes[1:], frames):
                for i in range(frames):
                    img = base_img.copy()
                    t = i / (frames - 1)
                    p = weighted_average(p0, p1, bezier_ease(t))
                    p = tuple(int(i) for i in p)
                    cv2.circle(img, center=p, radius=radius, color=color, thickness=-1)
                    cv2.circle(img, center=p, radius=radius, color=(0, 0, 0), thickness=1)
                    cv2.putText(img, color_key, (p[0] - 10, p[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
                    cv2.imshow("Steps", img)
                    cv2.waitKey(int(1000 / framerate))
                cv2.waitKey(int(1000 / framerate))

        print(len(solution), "steps")
        cv2.waitKey()


if __name__ == "__main__":
    cv2.namedWindow("Steps")
    cv2.namedWindow("screenshot")
    # cv2.setWindowProperty("screenshot", cv2.WND_PROP_TOPMOST, 1)
    # cv2.setWindowProperty("Steps", cv2.WND_PROP_TOPMOST, 1)
    cv2.moveWindow("Steps", 425, 0)
    cv2.moveWindow("screenshot", 425 * 2, 0)
    # run_test(15, "5-3")
    for i in range(5, 16):
        if i == 13:
            continue
        run_test(i)
