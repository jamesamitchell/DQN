from collections import deque
import random

class ExperienceReplay:

    def __init__(self, max_len):

        self.replays = deque(maxlen=max_len)

    def add(self, replay):

        self.replays.append(replay)

    def sample(self, batch_size):

        if len(self.replays) < batch_size:
            return random.sample(self.replays, len(self.replays))
        
        return random.sample(self.replays, batch_size)