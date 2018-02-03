from __future__ import division
import logging
import gym
import argparse
import os
from src.constants import Constants
from src.environment import TrainingEnvironment
from src.models import DQN
from src.processor_registry import ProcessorRegistry
from src.replay_memory import ReplayMemory
from gym import wrappers

parser = argparse.ArgumentParser(description='Setup the Google DeepMind Atari Paper (Nature).')
parser.add_argument('--frame-skip', default=4, type=int,
                    help='Number of frames to skip coming out of the Atari emulator.')
parser.add_argument('--null-op-max', default=30, type=int,
                    help='Maximum number of null operations after an Atari environment reset.')
parser.add_argument('--phi-length', default=4, type=int,
                    help='Number of consecutive frames to stack to pass to the model (considering frame skip).')
parser.add_argument('--memory', default=100000, type=int,
                    help='Replay memory size. This can lead to allocation errors if it is too large.')
parser.add_argument('--model-def', default='nature',
                    help="""Name assigned to the ModelDef class as given to the ModelRegistry class. The ModelDef
                            class defined the neural network to be used during training/play.""")
parser.add_argument('--minibatch-size', default=32, type=int,
                    help='Number of examples per minibatch.')
parser.add_argument('--batches', default=50000, type=int,
                    help='Number of batches per epoch.')
parser.add_argument('--epochs', default=100, type=int,
                    help='Number of epochs.')
parser.add_argument('--reward-clip', default=1, type=int,
                    help='Value to clip cumulative reward per state to in absolute value..')
parser.add_argument('--learning-rate', default=.00025, type=float,
                    help='Learning rate for training.')
parser.add_argument('--rho', default=.95, type=float,
                    help='Rho parameter for RMSprop.')
parser.add_argument('--discount', default=.99, type=float,
                    help='Discount (gamma) for future rewards.')
parser.add_argument('--fixed-update', default=10000, type=int,
                    help='Number of batch updates before the fixed network\'s weights are updated.')
parser.add_argument('--watch', action='store_true',
                    help='Turn off learning and watch network play.')
parser.add_argument('--no-consecutive-max', action='store_true',
                    help='Turn off the consecutive max for frames from the emulator.')
parser.add_argument('--load-model', default=None,
                    help='Path to a Keras model file.')
parser.add_argument('--game', default='PongNoFrameskip-v4',
                    help='Name of the Atari rom to load into opengym. (The default frame skip assumes you use a NoFrameskip version.)')
parser.add_argument('--processor', default='pong',
                    help="""Name of the Processor class found in the ProcessorRegistry
                         to use to process output from the emulator.""")
parser.add_argument('--save-path', default='dqn_checkpoint',
                    help='Folder to save training checkpoints.')
parser.add_argument('--annealing-frames', default=1000000, type=int,
                    help='Number of frames to anneal from 1 to the min epsilon.')
parser.add_argument('--min-epsilon', default=.1, type=float,
                    help='Minimum epsilon to anneal to.')
parser.add_argument('--gradient-clip', default=1., type=float,
                    help='Value to clip the gradient to, corresponds to setting the loss to |x| outside of the gradient clip zone. Set to a negative number to disable and use mean squared error.')
parser.add_argument('--max-epsilon', default=1, type=float,
                    help='Epsilon to start anneal.')
parser.add_argument('--play-epsilon', default=.05, type=float,
                    help='Epsilon to use when not training (during watch).')
parser.add_argument('--image-size', nargs='+', type=int, default=[80, 80],
                    help='Length and width of the processed image. (Multiple arguments, e.g.: 80 80)')
parser.add_argument('--actions', nargs='+', type=int, default=None,
                    help='Input the actions availble to the learner. Default is all available. (Multiple arguments, e.g.: 0 4 5)')
parser.add_argument('--log-file', default='log.txt',
                    help='Location of the file for log output.')
parser.add_argument('--monitor-path', default=None,
                    help='Location to output the video results the run.')

if __name__ == '__main__':
    # Parse inputs
    args = parser.parse_args()
    Constants.frame_skip = args.frame_skip
    Constants.null_op_max = args.null_op_max
    Constants.phi_length = args.phi_length
    Constants.replay_memory_size = args.memory if not args.watch else 1
    Constants.batches_per_epoch = args.batches
    Constants.minibatch_size = args.minibatch_size
    Constants.epochs = args.epochs
    Constants.learning_rate = args.learning_rate
    Constants.training = not args.watch
    Constants.fixed_update_count = args.fixed_update
    Constants.load_model_path = args.load_model
    Constants.save_path = args.save_path
    Constants.image_size = tuple(args.image_size)
    Constants.rho = args.rho
    Constants.game = args.game
    Constants.annealing_frames = args.annealing_frames
    Constants.play_epsilon = args.play_epsilon
    Constants.min_epsilon = args.min_epsilon
    Constants.discount = args.discount
    Constants.max_epsilon = args.max_epsilon
    Constants.reward_clip = args.reward_clip
    Constants.log_file = args.log_file
    Constants.model_def = args.model_def
    Constants.gradient_clip = args.gradient_clip
    Constants.consecutive_max = not args.no_consecutive_max
    Constants.processor = args.processor
    Constants.monitor_path = args.monitor_path


    # Derived Constants
    env = gym.make(Constants.game)
    actions = range(env.action_space.n) if args.actions is None else args.actions

    if Constants.monitor_path is not None:
        env = wrappers.Monitor(env, Constants.monitor_path)


    if not os.path.exists(Constants.save_path):
        os.makedirs(Constants.save_path)

    # Setup logging
    logger = logging.getLogger()
    log_to_file = True
    if log_to_file:
        fh = logging.FileHandler(os.path.join(Constants.save_path, Constants.log_file))
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    logger.setLevel(logging.INFO)

    logger.info(args)

    model = DQN(model_type=Constants.model_def,
                discount=Constants.discount,
                annealing_frames=Constants.annealing_frames,
                min_annealing_epsilon=Constants.min_epsilon,
                max_annealing_epsilon=Constants.max_epsilon,
                fixed_update_count=Constants.fixed_update_count,
                actions=actions,
                learning_rate=Constants.learning_rate,
                rho=Constants.rho,
                minibatch_size=Constants.minibatch_size,
                save_path=Constants.save_path,
                gradient_clip=Constants.gradient_clip,
                model_path=Constants.load_model_path)

    replay_memory = ReplayMemory(size=Constants.replay_memory_size,
                                 minibatch_size=Constants.minibatch_size,
                                 image_size=Constants.image_size,
                                 phi_length=Constants.phi_length)

    processor = ProcessorRegistry.get_processor(Constants.processor)

    gh = TrainingEnvironment(env=env,
                             model=model,
                             replay_memory=replay_memory,
                             input_processor=processor,
                             phi_length=Constants.phi_length,
                             frame_skip=Constants.frame_skip,
                             image_size=Constants.image_size,
                             training=Constants.training,
                             epochs=Constants.epochs,
                             batches_per_epoch=Constants.batches_per_epoch,
                             reward_clip=Constants.reward_clip,
                             null_op_max=Constants.null_op_max,
                             play_epsilon=Constants.play_epsilon,
                             minibatch_size=Constants.minibatch_size,
                             save_path=Constants.save_path,
                             consecutive_max=Constants.consecutive_max)

    try:
        gh.run()
    except OSError as e:
        raise Exception("If the error is a memory error, try shrinking the replay memory.", e)
    finally:
        env.close()