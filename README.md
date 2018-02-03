# Extensible Python DQN (ExPyDQN)

An extensible framework for training Deep Q Networks built on Keras and gym.

This framework was written as an extensible DQN implementation based on the Nature DeepMind paper. The code focuses
on logically compartmentalizing different components so that they may be easily understood. The neural network component
is based on the Keras library and adds functionality to it in the form of custom loss functions. The implementation aims 
to be memory efficient but the main goals were to be written in a clear, concise, and instructive manner.

## Installation

Installing via pip in a virtualenv is recommended using the requirements.txt file. You must also install libav-tools in order to use the Monitor class from OpenAI Gym.
 
```shell
pip install -r requirements.txt
```

The code was testing with the following configuration:
    - Kera 2.0.5
    - gym 0.9.2 (gym[atari]) 
    - atari-py 0.0.21
    - Theano 0.9.0

## Usage

The framework allows you to train and watch the model play from a simple interface. Output is logged to the --log-file
in the directory given to --save-path where the models are also saved. The ROM is selected via --game and the class
to process the observations (as defined in src/processor_registry.py) can be identified via --processor. The
neural network definition to use is given via --model-def (and defined in src/model_registry.py).

### Training

All parameters for runtime are given as command line arguments to the main.py file. A basic run can be accomplished via

```shell
python main.py --save-path example --memory 100000 --game PongNoFrameskip-v4
```

It is recommended to shrink the size of the replay memory via the --memory input to avoid allocation errors on most
machines.

### Watching

A model can be loaded (with the same parameter settings) and watched via the --watch input. This flag automatically
turns off training.

```shell
python main.py --load-model example/checkpoint_10.h5 --play-epsilon .05 --watch
```

#### Watching Monitor Results

The results will be saved to mp4, but the videos will be slowed down. Barring different settings, I would suggest something like this to speed them up to normal speed:

```shell
ffmpeg -i input.mkv -filter:v "setpts=0.25*PTS" output.mkv
```

### Extending the Framework

The two components that are able to be extended inside the framework are the neural network definitions and the
observation processing component. New models and image processors can be added via two aptly named registers. 
These registers allows you to easily extend the framework with your own classes representing new neural network 
configurations or screen processors.

#### Defining a Custom Neural Network

To define a custom neural network, create a new class extending ModelDef and decorate the class with the 
ModelRegistry.register_model decorator. Both classes are found in the src/model_registry.py file. 

The name assigned via ModelRegistry.register_model should be passed to the --model-def parameter.

You do not need to compile the model as the training is accomplished by a custom loss function defined in src/models.py.

#### Defining a Custom Processor

To define a custom processor for the output image of the emulator, create a new class extending the Processor class 
and decorate the class with the ProcessorRegistry.register_process decorator. Classes can be found in the 
src/processor_registry.py file.

The name assigned via ProcessorRegistry.register_process should be passed to the --processor parameter.

### Other Parameters

All of the parameters for training are exposed via the command line. This includes options like frame skipping, taking a
consecutive max, the null operation max, and phi length (number of frames to stack on top of each other to represent a
state). 

For a comprehensive list, please run the following command:

```shell
python main.py --help
```
 