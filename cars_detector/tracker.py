class Tracker:
    def __init__(self):
        self._next_id = 1

    def next_id(self) -> int:
        pid = self._next_id
        self._next_id += 1
        return pid

    def reset(self):
        self._next_id = 1
