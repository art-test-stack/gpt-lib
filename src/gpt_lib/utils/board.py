
class DummyBoard:
    def __init__(self) -> None:
        pass
    def log(self, *args, **kwargs) -> None:
        pass

class WandbBoard:
    def __init__(self, place: str = "wandb") -> None:
        import wandb
        self.place = place
        self.wandb = wandb

    def log(self, data: dict, step: int | None = None) -> None:
        if step is not None:
            self.wandb.log(data, step=step)
        else:
            self.wandb.log(data)
            
class Board:
    def __init__(self, place: str = "console") -> None:
        self.place = place
