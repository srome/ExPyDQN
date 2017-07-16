import numpy as np
import logging
import time

class TrainingEnvironment:
    """ Class to handle the processing of the game loop."""
    def __init__(self,
                 env,
                 model,
                 replay_memory,
                 input_processor,
                 phi_length,
                 frame_skip,
                 image_size,
                 training,
                 epochs,
                 batches_per_epoch,
                 reward_clip,
                 null_op_max,
                 play_epsilon,
                 minibatch_size,
                 save_path,
                 consecutive_max):
        self._env = env
        self.model = model
        self.replay_memory = replay_memory
        self._process_image = input_processor
        self.logger = logging.getLogger(__name__)
        self.phi_length = phi_length
        self.frame_skip = frame_skip
        self.image_size = image_size
        self.training = training
        self.epochs = epochs
        self.batches_per_epoch = batches_per_epoch
        self.reward_clip = reward_clip
        self.null_op_max = null_op_max
        self.play_epsilon = play_epsilon
        self.minibatch_size = minibatch_size
        self.save_path = save_path
        self.consecutive_max = consecutive_max

        self.reset_recent_frames()


    def run(self):
        for epoch in range(self.epochs):
            start = time.time()
            reward, episodes = self.run_epoch(num_steps=self.batches_per_epoch, training=self.training)

            # Get model info for reporting
            avg_loss, average_q = self.model.get_training_info()
            eps = self.model.get_epsilon()
            batches = self.model._batch_updates

            self.logger.info \
                ('Epoch: {epoch} | # Episodes: {episodes} |  # Batches: {batches} | Average Reward: {reward} | Average Q Value: {avgq} | Average Loss: {loss} | Epsilon: {epsilon} | Elapsed Time: {time} mins'.format(
                batches = batches, epoch=epoch, reward=reward, loss=avg_loss, epsilon=eps, episodes=episodes, time=(time.time() - start ) /60., avgq = average_q))
            if self.training:
                self.save_checkpoint(epoch=epoch)

    def reset_recent_frames(self):
        self._recent_frames = np.zeros(shape=(self.phi_length, self.image_size[0], self.image_size[1]), dtype=np.int8)

    def save_checkpoint(self, epoch):
        self.model.save(checkpoint=epoch, model_path = self.save_path)

    def render(self):
        self._env.render()

    def run_episode(self, training=True):
        state = self.reset_environment()
        is_terminal = False
        total_reward = 0
        steps = 0

        while not is_terminal:

            if not training:
                self.render()

            action = self.get_action(state, training)

            next_state, reward, is_terminal = self.step(action)

            if training:
                clipped_reward = np.clip(reward ,-self.reward_clip ,self.reward_clip)
                self.replay_memory.add_example(state=state,
                                               future_state=next_state,
                                               action=action,
                                               reward=clipped_reward,
                                               is_terminal=is_terminal)
                self.training_step()

            state = next_state
            total_reward += reward
            steps += 1

        return total_reward, steps

    def run_epoch(self, num_steps, training=True):
        """ Run a training epoch for a giving number of steps.
            Return the average reward, and number of episodes."""
        total_steps = 0
        total_reward = 0
        num_episodes = 0
        while total_steps < num_steps:
            reward, steps = self.run_episode(training=training)
            total_reward += reward
            total_steps += steps
            num_episodes += 1

        return total_reward / num_episodes, num_episodes

    def reset_environment(self):
        obs = self._env.reset()
        self.reset_recent_frames()

        # Perform a null operation to make game stochastic
        for k in range(np.random.randint(0 ,self.null_op_max, size=1)):
            obs, _, _, _ = self._env.step(0)

        self.store_observation(obs)
        return self._get_current_state()

    def get_action(self, state, training):
        epsilon_override = None

        if not training:
            epsilon_override = self.play_epsilon

        return self.model.get_action(image=state, epsilon_override=epsilon_override)

    def training_step(self):
        if self.replay_memory.replay_full():
            states, future_states, actions, rewards, is_terminal_inds = self.replay_memory.sample \
                (size=self.minibatch_size)
            self.model.train(states=states, rewards=rewards, future_states=future_states, actions=actions, is_terminal=is_terminal_inds)

    def step(self, action):
        """ This relies on the fact that the underlying environment creates new images for each frame.
            By default, opengym uses atari_py's getScreenRGB2 which creates a new array for each frame."""

        total_reward = 0

        obs = None
        for k in range(self.frame_skip):
            last_obs = obs

            obs, reward, is_terminal, info = self._env.step(action)
            total_reward += reward

            if is_terminal:
                # End episode if is terminal
                if k == 0 and last_obs is None:
                    last_obs = obs
                break

        if self.consecutive_max and self.frame_skip > 1:
            obs = np.maximum(last_obs, obs)

        # Store observation
        self.store_observation(obs)

        return self._get_current_state(), total_reward, is_terminal

    def store_observation(self, observation):
        processed_observation = self._process_image(observation)
        for k in range(1 ,self.phi_length):
            self._recent_frames[ k -1 ,: ,:] = self._recent_frames[k ,: ,:]
        self._recent_frames[k ,: ,:] = processed_observation

    def _get_current_state(self):
        return self._recent_frames.copy() # Copy is important! Otherwise state will = next_state in the loop
