import os
import random
from dataclasses import dataclass, field
from enum import Enum

import cv2
import numpy as np

import search
from animation import parametric_ease


#
# def all_lt(a, b):
#     for i, j in zip(a, b):
#         if i >= j:
#             return False
#     return True
#
#
# def all_lteq(a, b):
#     for i, j in zip(a, b):
#         if i > j:
#             return False
#     return True
#
#
# def all_gt(a, b):
#     for i, j in zip(a, b):
#         if i <= j:
#             return False
#     return True
#
#
# def all_gteq(a, b):
#     for i, j in zip(a, b):
#         if i < j:
#             return False
#     return True
#
#
# def any_lt(a, b):
#     for i, j in zip(a, b):
#         if i < j:
#             return True
#     return False
#
#
# def any_gt(a, b):
#     for i, j in zip(a, b):
#         if i > j:
#             return True
#     return False


class Direction(Enum):
    UP = 0,
    DOWN = 1,
    LEFT = 2,
    RIGHT = 3,


TEXTURE_DIR = "textures"
TEXTURES = []

TEXTURE_SRC_SIZE = 512
BLOCK_SIZE = 64

for filename in os.listdir(TEXTURE_DIR):
    if not filename.endswith("png"):
        continue
    f = os.path.join(TEXTURE_DIR, filename)
    if not os.path.isfile(f):
        continue
    TEXTURES.append(cv2.imread(f))
    # cv2.imshow(f, TEXTURES[-1])


def putTextCentered(image, text, org, font, fontScale, color, thickness=1, lineType=cv2.LINE_8, bottomLeftOrigin=False):
    size, baseline = cv2.getTextSize(text, font, fontScale, thickness)
    cv2.putText(image, text, (int(org[0] - size[0] / 2), int(org[1] - size[1] / 2 + 2 * baseline)), font, fontScale,
                color, thickness, lineType, bottomLeftOrigin)


# class Block:
#     def init(self, w, h, x, y):
@dataclass(order=True)
class Block:
    width: int = field(init=True, repr=True)
    height: int = field(init=True, repr=True)
    x: int = field(init=True, repr=True)
    y: int = field(init=True, repr=True)
    id: int = field(init=True, repr=False)
    texture: np.array = None

    def __post_init__(self):
        if self.texture is None:
            random.seed(str(self) + str(self.id))
            material = random.choice(TEXTURES)
            rotation = random.choice([cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE, None])
            if rotation is None:
                material = np.copy(material)
            else:
                material = cv2.rotate(material, rotation)

            tx = random.randint(0, TEXTURE_SRC_SIZE // BLOCK_SIZE - self.width) * BLOCK_SIZE
            ty = random.randint(0, TEXTURE_SRC_SIZE // BLOCK_SIZE - self.height) * BLOCK_SIZE
            self.texture = material[ty:ty + BLOCK_SIZE * self.height, tx:tx + BLOCK_SIZE * self.width]
            putTextCentered(self.texture, str(self.id),
                            (self.width * BLOCK_SIZE // 2, self.height * BLOCK_SIZE // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.rectangle(self.texture, (-1, -1), self.texture.shape[:2][::-1], (0, 0, 0), thickness=5)

    def corners(self):
        x, y, w, h = (self.x, self.y, self.width, self.height)
        return (x, y), (x + w, y), (x, y + h), (x + w, y + h)

    def keypoints(self):
        return self.x, self.y, self.x + self.width, self.y + self.height

    def overlaps(self, other):
        ax1, ay1, ax2, ay2 = self.keypoints()
        bx1, by1, bx2, by2 = other.keypoints()
        return ax1 < bx2 and ax2 > bx1 and ay1 < by2 and ay2 > by1

    def move(self, direction, distance=1):
        dx = 0
        dy = 0
        if direction == Direction.UP:
            dy = -distance
        elif direction == Direction.DOWN:
            dy = distance
        elif direction == Direction.LEFT:
            dx = -distance
        elif direction == Direction.RIGHT:
            dx = distance
        else:
            raise ValueError(f"Invalid direction {direction}")

        new_block = Block(self.width, self.height, self.x + dx, self.y + dy, self.id, self.texture)
        return new_block

    def __str__(self):
        return f'{self.width}x{self.height} @ ({self.x},{self.y})'

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(str(self))

    def draw(self, canvas):
        edge = 2
        x = self.x * BLOCK_SIZE
        y = self.y * BLOCK_SIZE
        h, w, d = self.texture.shape
        canvas[int(y + edge):int(y + h - edge), int(x + edge):int(x + w - edge)] = self.texture[edge:-edge, edge:-edge]
        return canvas


class BlockSlideGame:
    def __init__(self, size, blocks, goals, validate=True):
        self.width, self.height = size
        self.blocks = blocks
        self.goals = goals
        block_list = list(blocks.values())
        if validate:
            for i in range(len(block_list)):
                for j in range(i + 1, len(block_list)):
                    assert not block_list[i].overlaps(block_list[j]), \
                        f'Block {block_list[i]} overlaps with {block_list[j]}'

    def generateSuccessors(self):
        successors = []
        for block_id, block in self.blocks.items():
            # other_blocks = self.blocks[:i] + self.blocks[i + 1:]

            # Move each block in each direction
            for direction in Direction:
                valid = True
                count = 1
                new_block = block
                while valid:
                    new_block = new_block.move(direction)
                    x1, y1, x2, y2 = new_block.keypoints()

                    # Check new block position is out of bounds
                    if x1 < 0 or y1 < 0 or x2 > self.width or y2 > self.height:
                        break

                    # Check new block position overlaps another block
                    for b in self.blocks.values():
                        if b.id != block_id and new_block.overlaps(b):
                            valid = False
                            break
                    if valid:
                        move = (block.id, direction, count)
                        successors.append((self.do_move(*move, validate=False), move, 1))
                        count = count + 1

        return successors

    def do_move(self, block_id, direction, distance, validate=True):
        new_blocks = self.blocks.copy()
        new_blocks[block_id] = self.blocks[block_id].move(direction, distance)
        return BlockSlideGame((self.width, self.height), new_blocks, self.goals, validate)

    def isWinState(self):
        for block_id, x, y in self.goals:
            block = self.blocks[block_id]
            if block.x != x or block.y != y:
                return False
        return True

    def __str__(self):
        goal_block_list = [self.blocks[block_id] for block_id, x, y in self.goals]
        other_block_list = [block for block in self.blocks.values() if block not in goal_block_list]
        other_block_list.sort()
        return f'{self.width}x{self.height}|{goal_block_list}|{other_block_list}'

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return hash(self) == hash(other)

    def draw(self, canvas, exclude=(None,)):
        for block in self.blocks.values():
            if block.id in exclude:
                continue
            block.draw(canvas)
        return canvas

    def create_canvas(self):
        # img = np.zeros([self.height * BLOCK_SIZE, self.width * BLOCK_SIZE, 3], dtype=np.uint8)
        # img.fill(255)
        img = cv2.rotate(TEXTURES[-1], cv2.ROTATE_90_CLOCKWISE)[0:self.height * BLOCK_SIZE, 0:self.width * BLOCK_SIZE]
        (h, s, v) = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        s = s // 2
        v = v // 2
        return cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)


class BlockSlideSearch(search.SearchProblem):
    def __init__(self, start_state):
        self.start_state = start_state

    def getStartState(self):
        return self.start_state

    def isGoalState(self, state):
        return state.isWinState()

    def getSuccessors(self, state):
        return state.generateSuccessors()

    def getCostOfActions(self, actions):
        return len(actions)


if __name__ == "__main__":
    blocks = (
        Block(2, 2, 0, 0, 1),
        Block(1, 1, 2, 0, 2),
        Block(1, 1, 2, 1, 3),
        Block(2, 1, 3, 0, 4),
        Block(2, 1, 3, 1, 5),
        Block(1, 2, 0, 2, 6),
        Block(1, 2, 1, 2, 7),
        Block(1, 2, 3, 2, 8),
        Block(1, 2, 4, 2, 9),
    )
    start_state = BlockSlideGame((5, 4), {block.id: block for block in blocks}, ((1, 3, 0),))
    print(start_state)

    b = start_state.blocks

    problem = BlockSlideSearch(start_state)
    solution = search.astar(problem)
    state = start_state
    blank_canvas = state.create_canvas()

    window_name = "Solution"

    cv2.imshow(window_name, state.draw(blank_canvas.copy()))
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
    cv2.waitKey(10000)

    frames_per_move = 8
    framerate = 60

    for i, (block_id, direction, distance) in enumerate(solution):
        print(f"{i}:\t{block_id} {direction.name} {distance}")
        canvas = state.draw(blank_canvas.copy(), exclude=(block_id,))
        block = state.blocks[block_id]

        for t in range(frames_per_move):
            d = distance * parametric_ease(t / frames_per_move)
            frame = block.move(direction, distance=d).draw(canvas.copy())
            cv2.imshow(window_name, frame)
            cv2.waitKey(int(1000 / framerate))

        state = state.do_move(block_id, direction, distance)
        cv2.imshow(window_name, state.draw(blank_canvas.copy()))
        cv2.waitKey(int(1000 / framerate))

    cv2.waitKey(5000)
