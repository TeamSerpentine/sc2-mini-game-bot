# Serpentine SC II Bot - "Danger Noodle"

The serpentine Starcraft II bot has a hierarchical structure based on hand built
policies which correspond to actions an agent is able to perform. A deep 
neural network is used to map observations to policies. The agent trains against
scripted agents and is rewarded for increasing it's own score. By iterating 
training cycles, the network maps inputs to outputs while maximizing the score.

Also have a look at [the technical report](https://serpentineai.nl/publications) for a more in depth description of our approach.

## The network

The network is built up of several fully connected layers. It takes input from
a pre-processing system called the unit tracker, and outputs expected rewards
for the policies that the agent can perform. In essence the network is trying to
learn the mapping from observation to a policies expected reward as accurately as possible, and we
simply instruct it to always choose the best option available to it. We always try to perform the policy
with the highest expected reward, however when this is not possible we simply
default to another policy until the agent finds one that does work.

Because the network models the game, it is possible for it to learn without 
actually winning in each training session. The more accurate the model, 
the better we expect our performance to be.

The input to the network does not only include the output of the unit tracker for
the current time step but also for a number of previous time steps. This way
we try to encode temporal information into the network.

## Pre processing

The pre processing unit is called the unit tracker. It reads the PySC2 API
and constructs a number of useful metrics for the game including but not
limited to: Friendly and enemy unit coordinates, death flags, spawn timers and
pineapple spawn timers. It is a compact representation of the information that
can be gained from the API. This pre processing step allows us to have smaller 
networks, and we are able to do away with any convolutional layers that might
be necessary when processing inputs as image data.

## Policies

Policies are low level actions that agents can perform. They often correspond to
API action calls to the Starcraft interface with hardcoded parameters. They
can also be a simple combination of action calls. Examples include "Move camera to pineapple",
"Attack nearest enemy" and "Walk to left flank". The total number of policies is
21, but we do not always include all policies during training. The neural network
outputs a predicted reward for each of these policies, and we choose the one that
is expected to yield the highest score. 

## Training

The network is trained by iterating over many cycles of gameplay. The agent is trained against a version
of itself and against an aggressive scripted agent. This way we can evaluate performance
over time against itself and we have a baseline reading against the scripted agent.
We train the agent against both option to prevent overfitting the network
on a limited game state space. Learning rates are typically varied. 

At the start of each
training cycle we introduce random actions. We typically start at 80% randomness and
build down to 10%. This is called an "epsilon greedy policy". This increases the
exploration rate of the action space.


## Setup for running

* Unzip the file "hand in"
* The call to the agent will be relative to the path to that folder in the terminal

```text
python -m pysc2.bin.agent --map CompetitionMap --agent handin.danger_noodle.Serpentine --use_feature_units
```

Side note, there is code in the ' handin/strategy/serpentine.py' that makes sure 
that only a part of the GPU is used, otherwise it may cause problem when running 
multiple models on one machine. You can disbale this by setting the 'multiple_models' to False.

Calling the agent against an opponent depends on the side of the map, if we are the first agent use

```text
--agent handin.danger_noodle.Serpentine
```
otherwise use

```text
--agent2 handin.danger_noodle.SerpentineV2
```
This is required for indicating the start positions.
