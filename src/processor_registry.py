import numpy as np

class ProcessorRegistry:
    """
    Class to store functions which process emulator output.
    """
    registry = {}

    @classmethod
    def register_processor(cls, name):
        def decorator(f):
            cls.registry[name] = f()
            return f
        return decorator

    @classmethod
    def get_processor(cls, name):
        return cls.registry[name]


class Processor:
    def __call__(self, observation):
        pass


@ProcessorRegistry.register_processor('pong')
class PongProcessor(Processor):
    def __call__(self, observation):
        # Grayscale, downsample, and crop
        temp = np.zeros((observation.shape[0], observation.shape[1]), dtype=np.int8)  # Blank array for new image

        # Luminosity grayscale
        temp[:, :] += (.2126 * observation[:, :, 0]).astype(np.int8)
        temp[:, :] += (.7156 * observation[:, :, 1]).astype(np.int8)
        temp[:, :] += (.0722 * observation[:, :, 2]).astype(np.int8)

        # Downsample
        temp = temp[::2, ::2]

        # Crop
        return temp[17:-8, :]