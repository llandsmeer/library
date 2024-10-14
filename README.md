# <NAME>: A Library for Realistic Embodied Brain Simulations

At the core, a closed-loop realistic embodied brain simulation must contain:

 -  Neurons, modelled at the level of scale that allows modelling different cell types and anatomy
 -  Their connections, including their net effect and delays
 -  Muscle model, i.e. Hill-type
 -  Body model containing joints and muscle connections
 -  Physics engine and environmental interaction
 -  Sensory feedback in the form of proprioceptive and sensory feedback


In a real-world use-case, we also want

 - Massive parallel execution on modern accelerator hardware
 - Ability to apply higher-order optimisation algorithms - i.e. gradient descent

 # Authors

  - Joy Brand
  - Lennart P. L. Landsmeer


