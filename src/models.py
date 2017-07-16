import logging
import numpy as np
from keras.models import Sequential
from keras.models import load_model, Model
import keras.backend as K
from keras.layers import Lambda, Input, Dense, Flatten, Conv2D
from copy import deepcopy
import keras
from src.model_registry import ModelRegistry


def process_input(cast_to_type, variables):
    """ Scales the inputs specified between 0,1.
        Expects numpy arrays to be in the keyword arguments.
        Only for use on public functions."""
    def decorator(f):
        def wrapper_function(*args, **kwargs):
            for var_name in variables:
                try:
                    if kwargs[var_name].dtype != cast_to_type:
                        kwargs[var_name] = kwargs[var_name].astype(cast_to_type)
                    kwargs[var_name] = kwargs[var_name]/255. # Replace with scaled version
                except KeyError:
                    raise Exception('{var} must be called via a keyword argument.'.format(var=var_name))
            return f(*args, **kwargs)
        return wrapper_function
    return decorator


class DQN:
    """
    DQN framework for Reinforcement Learning.

    Interface for deep learning: get_action, train, save.
    All public functions expect any state inputs to be already processed, int8 images.
    get_action and train use a public facing process_input decorator function to cast to float32 and scale the inputs.
    """

    def __init__(self,
                 model_type,
                 discount,
                 annealing_frames,
                 min_annealing_epsilon,
                 max_annealing_epsilon,
                 fixed_update_count,
                 actions,
                 learning_rate,
                 rho,
                 minibatch_size,
                 save_path,
                 gradient_clip,
                 model_path=None):
        self._discount = discount
        self._anneal = annealing_frames
        self._counter = 0
        self._min_annealing_epsilon = min_annealing_epsilon
        self._max_annleaing_epsilon = 1 if (max_annealing_epsilon > 1) or (
        max_annealing_epsilon < 0) else max_annealing_epsilon
        self._fixed_model_update = fixed_update_count
        self._num_actions = len(actions)
        self._logger = logging.getLogger(__name__)
        self._learning_rate = learning_rate
        self._rho = rho
        self._minibatch = minibatch_size
        self._save_path = save_path
        self._gradient_clip = gradient_clip

        self._loss_update_count = 5000
        self._loss_counter = 0
        self._losses = np.array([np.nan] * self._loss_update_count)
        self._average_q = np.array([np.nan] * self._loss_update_count)

        # Create two models, one for the fixed target network, and one for the learner
        if model_path is not None:
            self._model = load_model(model_path)
            self._fixed_model = load_model(model_path)
        else:
            model_build_function = ModelRegistry.get_model(model_type)
            self._model, self._fixed_model = model_build_function(self._num_actions), model_build_function(
                self._num_actions)
            self._update_fixed_model()  # Fixed model = model at start

        # Initialize
        self._actions = np.array(actions)
        self._batch_updates = 0
        self._build_training_model()

        # Must be zeros
        self._dummy_y_1 = np.zeros((self._minibatch, 1))
        self._dummy_y_2 = np.zeros((self._minibatch, self._num_actions))

    def _build_training_model(self):
        y_pred = self._model.output
        rewards = Input(name='future_rewards', shape=(1,))
        future_max_rewards = Input(name='rewards', shape=(1,))
        output_mask = Input(name='output_mask', shape=(self._num_actions,))
        is_terminal = Input(name='is_terminal', shape=(1,))
        discount = self._discount

        # Based on deep_q_rl and keras-rl implementations
        def clipped_masked_error(args):
            y_pred, rewards, future_max_rewards, output_mask, is_terminal = args

            masked_y_pred = K.sum(y_pred * output_mask, axis=1).reshape((self._minibatch, 1))
            target = rewards + discount * (K.ones_like(is_terminal) - is_terminal) * future_max_rewards

            diff = masked_y_pred - target

            # Input masked y_pred and target into huber loss
            if self._gradient_clip > 0:
                error = K.abs(diff)  # batch size x 1
                quadratic_part = K.minimum(error, self._gradient_clip)
                unsummed_loss = 0.5 * K.square(quadratic_part) + self._gradient_clip * (error - quadratic_part)
                final_loss = K.mean(unsummed_loss)  # axis=-1 ends up being K.mean with the later trick in losses
            else:
                final_loss = K.mean(.5 * K.square(diff))

            return final_loss

        loss_out = Lambda(clipped_masked_error, output_shape=(1,), name='loss')(
            [y_pred, rewards, future_max_rewards, output_mask, is_terminal])
        ins = [self._model.input] if type(self._model.input) is not list else self._model.input

        trainable_model = Model(inputs=ins + [rewards, future_max_rewards, output_mask, is_terminal],
                                outputs=[loss_out, y_pred])

        losses = [
            lambda y_true, y_pred: y_pred,
            # Keras uses the sum of this list's functions as the "loss". y_pred here = loss_out above
            lambda y_true, y_pred: K.zeros_like(y_pred)
        ]

        trainable_model.compile(optimizer=keras.optimizers.RMSprop(lr=self._learning_rate, rho=self._rho), loss=losses)
        self.trainable_model = trainable_model

    def get_epsilon(self, epsilon_override=None):
        if epsilon_override is None:
            return 1. - (self._max_annleaing_epsilon - self._min_annealing_epsilon) * np.min(
                [self._batch_updates, self._anneal]) / self._anneal
        else:
            return epsilon_override

    @process_input(cast_to_type=np.float32, variables=['image'])
    def get_action(self, image=None, epsilon_override=None):
        epsilon = self.get_epsilon(epsilon_override)
        if np.random.uniform(0, 1) < 1 - epsilon:
            rewards = self._get_q_values(image=image)
            action = self._actions[np.argmax(rewards)]
        else:
            action = np.random.choice(self._actions)

        return action

    def _get_q_values(self, image):
        return self._get_model(fixed=False).predict(np.array([image], copy=False), batch_size=1)[0]

    @process_input(cast_to_type=np.float32, variables=['states', 'future_states'])
    def train(self, states, rewards, future_states, actions, is_terminal):
        future_max_rewards, output_mask, is_terminal = self._build_training_variables(future_states, actions,
                                                                                      is_terminal)

        # This next piece is a quirk from hijacking the loss function in Keras. We have to give y values even though
        # we will not use them during training.

        # ## Testing code to validate output.
        # ypred = np.sum(self.get_model(fixed=False).predict(states) * output_mask, axis=1).reshape((32,1))
        # target = rewards + self._discount * future_max_rewards * (1-is_terminal)
        # diff = np.abs(ypred - target)
        # qp = np.minimum(diff, np.ones(diff.shape))
        # lossa = .5*qp**2+(diff-qp)

        loss = self.trainable_model.train_on_batch(x=[states, rewards, future_max_rewards, output_mask, is_terminal],
                                                   y=[self._dummy_y_1, self._dummy_y_2])

        self._average_q[self._loss_counter % self._loss_update_count] = self._get_model(fixed=False).predict(
            states).mean(axis=1).mean()
        self._losses[self._loss_counter % self._loss_update_count] = loss[0]

        self._batch_updates += 1
        self._loss_counter += 1

        if self._batch_updates % self._fixed_model_update == 0:
            self._update_fixed_model()
            # self._logger.info('Updating fixed model weights.')

    def _build_training_variables(self, future_states, actions, is_terminal):
        """ Function assumes future_states has already been processed. """
        future_rewards = self._get_model(fixed=True).predict(future_states)
        training_mask = np.zeros(future_rewards.shape)
        max_rewards = np.max(future_rewards, axis=1).reshape((len(future_rewards), 1))
        action_indexes = np.array([[np.where(self._actions == action)[0][0]] for action in actions])
        is_terminal = is_terminal * 1.  # convert to float

        # Update reward for each action
        # Relies on the fact that the train function has other actions zeroed out
        for index0, index1 in zip(range(self._minibatch), action_indexes):
            training_mask[index0, index1] = 1.

        return max_rewards, training_mask, is_terminal

    def get_training_info(self):
        try:
            mean_val = np.nanmean(self._losses)
            avg_q = np.nanmean(self._average_q)
        except:
            mean_val = None
            avg_q = None
        self._losses *= np.nan
        self._average_q *= np.nan
        self._loss_counter = 0
        return mean_val, avg_q

    def save(self, checkpoint=0, model_path=''):
        self._get_model(fixed=False).save(os.path.join(model_path, 'checkpoint_%s.h5' % checkpoint))

    def _get_model(self, fixed=False):
        if fixed:
            return self._fixed_model
        return self._model

    def _update_fixed_model(self):
        for k in range(len(self._model.layers)):
            new_weights = deepcopy(self._model.layers[k].get_weights())
            self._fixed_model.layers[k].set_weights(new_weights)  # guarantee the fixed model sees different values
