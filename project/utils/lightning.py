import pytorch_lightning as pl


class TrackTestAccuracyCallback(pl.callbacks.Callback):
    def __init__(self, datamodule) -> None:
        super().__init__()
        self.datamodule = datamodule

    def on_validation_end(self, trainer, module):
        trainer.test(model=module, verbose=False, datamodule=self.datamodule)
