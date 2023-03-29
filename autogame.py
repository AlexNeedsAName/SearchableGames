import time
import cv2
import pyautogui

top = 53
left = 0


def crop(image, rect):
    x, y, w, h = rect
    return image[y:y + h, x:x + w]


def scale(image, scale, inter=cv2.INTER_AREA):
    h, w = image.shape[:2]
    dim = (int(h * scale), int(w * scale))

def drag(point1, point2, duration=0.1):
    x, y = point1
    pyautogui.moveTo(x+left, y+top)
    x, y = point2
    pyautogui.dragTo(x+left, y+top, duration*5, button='left')
    pyautogui.sleep(duration)
    pyautogui.moveTo(1,1)
    pyautogui.sleep(duration)


def click(point, duration=0.1):
    x, y = point
    # print("Clicking ({}, {}) for {}s".format(x, y, duration))
    pyautogui.moveTo(x + left, y + top)
    pyautogui.mouseDown()
    pyautogui.sleep(duration)
    pyautogui.mouseUp()
    pyautogui.moveTo(1,1)


def click_transition(state, point, duration=0.1):
    if state.draw is not None:
        cv2.circle(state.draw, point, 8, (255, 255, 0), -1)
        cv2.imshow("test", state.draw)
        cv2.waitKey(1)
    click(point, duration)
    return True


def type_text_transition(text):
    def do_the_typing(state, point):
        pyautogui.typewrite(text)
        return click_transition(state, point)

    return do_the_typing


def offset_click_transition(x, y):
    def do_the_click(state, point):
        return click_transition(state, (point[0] + x, point[1] + y))

    return do_the_click


def nothing(_state):
    return True


def no_preprocess(state):
    return state.image


class State:

    def __init__(self, name, on_process=nothing):
        self.name = name
        self.transitions = []
        self.image = None
        self.draw = None
        self.raw = None
        self.on_process = on_process
        self.time = None
        self.can_transition = True

    def process(self, image, draw=None, now=None):
        self.image = image
        self.draw = draw
        self.raw = draw.copy()
        if now is None:
            now = time.time()
        self.time = now
        cv2.putText(self.draw, "State: {}".format(self.name), (100, 762), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 0), 2, 2)
        self.can_transition = self.on_process(self)

    def add_transition(self, button, state, on_transition=click_transition, preprocess=no_preprocess):
        self.transitions.append((button, state, on_transition, preprocess))

    def go_to_next(self):
        if not self.can_transition:
            return self
        for button, next_state, transition_function, preprocess in self.transitions:
            location = button.find_in(preprocess(self), self.draw)
            if location is not None and transition_function(self, location):
                return next_state
        return self


class Debouncer:
    def __init__(self, size):
        self.size = size
        self.data = [None] * size
        self.index = 0

    def add(self, item):
        self.data[self.index] = item
        self.index = (self.index + 1) % self.size

    def get(self):
        value = self.data[0]
        for entry in self.data[1:]:
            if value != entry:
                return None
        return value

    def clear(self):
        self.data = [None] * self.size


class Button:
    def __init__(self, filename, confidence=0.98, debounce_size=1, search_area=None, flippable=False):
        self.template = cv2.cvtColor(cv2.imread("images/{}".format(filename)), cv2.COLOR_BGR2GRAY)
        self.mask = None
        self.filename = filename
        self.confidence = confidence
        self.position = Debouncer(debounce_size)
        self.search_area = search_area
        self.flippable = flippable

    def reload(self, filename):
        self.template = cv2.cvtColor(cv2.imread("images/{}".format(filename)), cv2.COLOR_BGR2GRAY)
        self.filename = filename

    def find_in(self, image, draw_on=None):
        if self.search_area is not None:
            image = crop(image, self.search_area)
        if self.template is None:
            return
        res = cv2.matchTemplate(image, self.template, cv2.TM_CCOEFF_NORMED, self.mask)
        min_val, max_val, _min_loc, max_loc = cv2.minMaxLoc(res)
        flipped = False
        if self.flippable:
            flipped_template = cv2.flip(self.template, 1)
            flipped_mask = cv2.flip(self.mask, 1)
            # cv2.imshow("item", flipped_template)
            res2 = cv2.matchTemplate(image, flipped_template, cv2.TM_CCOEFF_NORMED, flipped_mask)
            min_val2, max_val2, _min_loc2, max_loc2 = cv2.minMaxLoc(res2)
            # print(max_val2, max_val)
            if max_val2 > max_val:
                # print("Flipped matches better")
                min_val, max_val, _min_loc, max_loc = (min_val2, max_val2, _min_loc2, max_loc2)
                flipped = True
        if self.mask is not None:
            cv2.imshow("quest items", res)
        if max_val > self.confidence or self.mask is not None:
            h, w = self.template.shape[:2]
            top_left = (max_loc[0], max_loc[1])
            if self.search_area is not None:
                top_left = (top_left[0] + self.search_area[0], top_left[1] + self.search_area[1])
            lower_right = (top_left[0] + w, top_left[1] + h)
            point = tuple((min_dim + max_z) // 2 for min_dim, max_z in zip(top_left, lower_right))
            if draw_on is not None:
                cv2.rectangle(draw_on, top_left, lower_right, (255, 255, 0), 2)
                if self.mask is not None:
                    cv2.putText(draw_on, "{:.2f}%".format(max_val * 100), top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (255, 255, 0), 2, 2)
            if self.mask is not None:
                offset = -int(w / 4)
                if flipped:
                    offset = -offset
                point = (point[0] + offset, point[1])
            self.position.add(point)
            point = self.position.get()
            if max_val > self.confidence:
                return point
        return None

    def presence_confidence(self, image, draw_on=None):
        res = cv2.matchTemplate(image, self.template, cv2.TM_CCOEFF_NORMED)
        _min_val, max_val, _min_loc, max_loc = cv2.minMaxLoc(res)
        h, w = self.template.shape[:2]
        top_left = max_loc
        lower_right = (top_left[0] + w, top_left[1] + h)
        if draw_on is not None:
            cv2.rectangle(draw_on, top_left, lower_right, (255, 255, 0), 2)
        return max_val

    def is_present(self, image, draw_on=None):
        return self.presence_confidence(image, draw_on) > self.confidence
