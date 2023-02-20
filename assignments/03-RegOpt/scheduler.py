from typing import List

from torch.optim.lr_scheduler import _LRScheduler

import math


class CustomLRScheduler(_LRScheduler):
    """
    class for the scheduler
    """

    def __init__(self, optimizer, T_max, last_epoch=-1, eta_min=0, verbose=False):
        """
        initialize scheduler
        """
        self.T_max = T_max
        self.eta_min = eta_min

        self._last_restart: int = 0
        self._cycle_counter: int = 0
        self._init: bool = False

        super(CustomLRScheduler, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> List[float]:
        """
        cosine scheduler
        """
        if self._init is False:
            self._init = True
            return self.base_lrs

        step = self.last_epoch + 1
        self._cycle_counter = step - self._last_restart

        lrs = [
            (
                self.eta_min
                + ((lr - self.eta_min) / 2)
                * (
                    math.cos(
                        math.pi * ((self._cycle_counter) % self.T_max) / self.T_max
                    )
                    + 1
                )
            )
            for lr in self.base_lrs
        ]

        if self._cycle_counter % self.T_max == 0:
            # Adjust the cycle length.
            self._cycle_counter = 0
            self._last_restart = step

        return lrs
