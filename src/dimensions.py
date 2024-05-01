from dataclasses import dataclass

@dataclass
class Dimensions:
    batch_size: int
    frame_count: int
    height: int
    width: int
    channels: int

    @property
    def shape(self):
        return (self.batch_size, self.frame_count, self.height, self.width, self.channels)

    @property
    def example_shape(self):
        return (self.frame_count, self.height, self.width, self.channels)

