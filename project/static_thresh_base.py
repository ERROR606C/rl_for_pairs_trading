from env import Actions

class ZScoreBaseline:
    def __init__(self, entry_threshold=2.0, exit_threshold=0.5):
        self._entry = entry_threshold
        self._exit = exit_threshold

    def predict(self, obs):
        zscore = abs(obs["spread_z_score"][0])  # abs since we don't distinguish long/short yet
        in_position = obs["position"]

        if in_position == 0 and zscore > self._entry:
            return Actions.Swap.value  # enter
        elif in_position == 1 and zscore < self._exit:
            return Actions.Swap.value  # exit
        else:
            return Actions.Maintain.value