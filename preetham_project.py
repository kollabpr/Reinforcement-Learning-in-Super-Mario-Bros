import time
begin = time.time() # gives the current time 

import os # to create new directory
import numpy
import gym # to import the wwrapper classes 
import torch # to import Pytorch 
import gym_super_mario_bros # to bild the game environment

from PIL import Image as img # to create images/gifs

# Wrapper Classes
from gym.spaces import Box 
from gym.wrappers import FrameStack
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT # simple moves for the ai agent

from torchvision import transforms # to transform the images

from stable_baselines3 import PPO # The reinforcement learning algorithm
from stable_baselines3.common.callbacks import BaseCallback # callbacks to train the model

# To perform Frame Skipping on the Super Mario Environment
class Frames_Skipping(gym.Wrapper):
  def __init__(self, env, skip):
      super().__init__(env)
      self._skip = skip

  def step(self, ai_action_space):
      rwd_func = 0.0
      finish = False
      for i in range(self._skip):
          observation, reward, finish, data = self.env.step(ai_action_space)
          rwd_func += reward
          if finish:
              break
      return observation, rwd_func, finish, data

#To perform Compression on the Super Mario Environment
class Compress(gym.ObservationWrapper):
  def __init__(self, env, shape):
      super().__init__(env)
      if isinstance(shape, int):
          self.shape = (shape, shape)
      else:
          self.shape = tuple(shape)
      self.observation_space = Box(low=0, high=255, shape = self.shape + self.observation_space.shape[2:], dtype=numpy.uint8)

  def observation(self, observation):
      my_transforms = transforms.Compose([transforms.Resize(self.shape), transforms.Normalize(0, 255)])
      return my_transforms(observation).squeeze(0)

# To perform greyscaling on the Super Mario Environment
class convert_to_grey(gym.ObservationWrapper):
  def __init__(self, env):
      super().__init__(env)
      self.observation_space = Box(low=0, high=255, shape=self.observation_space.shape[:2], dtype=numpy.uint8)

  def change_alignment(self, observation):
      observation = numpy.transpose(observation, (2, 0, 1))
      return torch.tensor(observation.copy(), dtype=torch.float)

  def observation(self, observation):
      transform = transforms.Grayscale() 
      return transform(self.change_alignment(observation))

# To train the RL agent
class TrainAndLoggingCallback(BaseCallback):
  def __init__(self, check_freq, save_path, verbose=1):
      super(TrainAndLoggingCallback, self).__init__(verbose)
      self.check_freq = check_freq
      self.save_path = save_path

  def _init_callback(self):
      if self.save_path is not None:
          os.makedirs(self.save_path, exist_ok=True)

  def _on_step(self):
      if self.n_calls % self.check_freq == 0:
          print("Trained the model for",self.n_calls," episodes.")
          model_path = os.path.join(self.save_path, 'training_model_{}'.format(self.n_calls))
          self.model.save(model_path)
      return True  

# To save the best run as a gif file
def save_gif(model, image_file, begin):
    dream_run = []
    reward_list = []
    best_reward = 0

    for i in range(20):
      env = build_env()
      images = [img.fromarray(env.render(mode='rgb_array'))]
      observation = env.reset()
      max_reward = 0

      for i in range(0, 2000):
        b = torch.Tensor(4, 84, 84)
        torch.stack(observation._frames, out=b)
        ai_action_space, dummy = model.predict(b.numpy())
        observation, reward, finish, dummy = env.step(ai_action_space)
        max_reward = max_reward + reward

        if i % 2 == 0:
          images.append(img.fromarray(env.render(mode='rgb_array')))
        if finish:
          break

      reward_list.append(max_reward)

      if max_reward > best_reward or (max_reward == best_reward and len(images) > len(dream_run)):
        best_reward = max_reward
        dream_run = images
    dream_run[0].save(image_file, save_all=True, append_images=dream_run[1:], loop=0, duration=1)
    avg_reward= sum(reward_list) / len(reward_list)

    print("--------------------------------------------------------")
    print("AVG reward of 900000 episodes is: ",avg_reward)
    print("created gif file: ", image_file)
    print("900000 episodes Completed in %s seconds" % (time.time() - begin))

# Creating the game environment
def build_env():
  make_environment = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
  frame_skip_environment = Frames_Skipping(make_environment, skip=4)
  greyscale_environment = convert_to_grey(frame_skip_environment)
  compressed_environment = Compress(greyscale_environment, shape=84)
  frame_stacked_environment = FrameStack(compressed_environment, num_stack=4)
  game = JoypadSpace(frame_stacked_environment, SIMPLE_MOVEMENT)
  return game

# Create a model directory
model_gif = './model/'
if model_gif is not None:
    os.makedirs(model_gif, exist_ok=True)

# Core logic of the program
model = PPO('CnnPolicy', build_env())
model.learn(total_timesteps=900000, callback=TrainAndLoggingCallback(check_freq = 100, save_path = './train/')) # Running the code for 9 million episodes 
save_gif(model, f"/Users/preetham/Downloads/ML_Project/model/model.gif",begin) 