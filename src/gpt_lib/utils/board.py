from typing import Literal

try:
    import wandb
except:
    wandb = None
try:
    import tensorboard as tb
except:
    tb = None


class DummyBoard:
    def __init__(self) -> None:
        pass
    def log(self, *args, **kwargs) -> None:
        pass

class WandbBoard:
    def __init__(self, place: str = "wandb") -> None:
        self.place = place
        # self.wandb = wandb

    def log(self, data: dict, step: int | None = None) -> None:
        if step is not None:
            wandb.log(data, step=step)
        else:
            wandb.log(data)

class TensorBoard:
    def __init__(self):
        _board = tb.SummaryWrite()

    def log(self, *args, **kwargs):
        pass

import os
rank = os.getenv("RANK", -1)

_available_boards = {"dummy": DummyBoard}
if rank < 1:
    if tb is not None:
        _available_boards["tensorboard"] = TensorBoard
    if wandb is not None:
        _available_boards["wandb"] = WandbBoard
    
class Board:
    def __init__(self, place: str = "dummy", *args, **kwargs) -> None:
        self._board = _available_boards.get(place, "dummy")(*args, **kwargs)

    def log(self, data: dict, step: int | None = None) -> None:
        self._board(data, step)
        # if self.place == "console":
        #     if step is not None:
        #         print(f"Step {step}: ", data)
        #     else:
        #         print(data)
        # else:
        #     raise NotImplementedError(f"Board logging not implemented for place: {self.place}")
