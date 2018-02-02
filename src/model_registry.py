from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Conv2D
import keras

class ModelRegistry:
    """
    Class to store functions which instantiate neural networks.
    """
    registry = {}

    @classmethod
    def register_model(cls, name):
        def decorator(f):
            cls.registry[name] = f()
            return f
        return decorator

    @classmethod
    def get_model(cls, name):
        return cls.registry[name]

class ModelDef:

    def __call__(self, num_actions):
        """ Function to implement to define neural network."""
        pass

@ModelRegistry.register_model('nature')
class NatureModel(ModelDef):
    """ Class to instantiate and return the neural network from the DeepMind Nature paper."""

    def __call__(self, num_actions):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(8, 8),
                         activation='relu',
                         strides=4,
                         input_shape=(4, 80, 80),
                         data_format='channels_first', padding='same',
                         kernel_initializer=keras.initializers.glorot_uniform(),
                         bias_initializer=keras.initializers.constant(value=.1)))
        model.add(Conv2D(64,
                         (4, 4),
                         activation='relu',
                         data_format='channels_first', padding='same',
                         kernel_initializer=keras.initializers.glorot_uniform(),
                         bias_initializer=keras.initializers.constant(value=.1),
                         strides=2))
        model.add(Conv2D(64,
                         (3, 3), strides=1,
                         activation='relu',
                         data_format='channels_first', padding='same',
                         kernel_initializer=keras.initializers.glorot_uniform(),
                         bias_initializer=keras.initializers.constant(value=.1)))
        model.add(Flatten())
        model.add(Dense(512, activation='relu',
                        kernel_initializer=keras.initializers.glorot_uniform(),
                        bias_initializer=keras.initializers.constant(value=.1)))
        model.add(Dense(num_actions,
                        activation='linear',
                        kernel_initializer=keras.initializers.glorot_uniform(),
                        bias_initializer=keras.initializers.constant(value=.1)))
        return model
