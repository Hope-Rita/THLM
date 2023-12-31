class EarlyStopMonitor(object):
    def __init__(self, max_round=3, higher_better=True, tolerance=1e-6):
        self.max_round = max_round
        self.num_round = 0

        self.epoch_count = 0
        self.best_epoch = 0

        self.last_best = 0
        self.higher_better = higher_better
        self.tolerance = tolerance

    def early_stop_check(self, curr_val):
        isImprove = False
        print(
            "cur metric:{}, last metric:{}, tolerance:{}".format(curr_val, self.last_best, (curr_val - self.last_best)))
        print(f"improve:{(curr_val - self.last_best) > self.tolerance}")
        if not self.higher_better:
            curr_val *= -1
        if (curr_val - self.last_best) > self.tolerance:
            isImprove = True
            self.last_best = curr_val
            self.num_round = 0
            self.best_epoch = self.epoch_count
        else:
            self.num_round += 1

        self.epoch_count += 1


        return self.num_round >= self.max_round, isImprove