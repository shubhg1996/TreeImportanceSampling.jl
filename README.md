# TreeImportanceSampling.jl

Generate importance weighted samples via stochastic Tree Search

## Overview

**TreeImportanceSampling.jl** is a Julia package that implements a stochastic tree search algorithm for generating importance weighted samples. This powerful tool is designed to help researchers and practitioners efficiently estimate probabilities in complex stochastic models, particularly in scenarios where rare events matter. The package is built entirely in Julia to ensure high performance and ease-of-integration with other scientific computing workflows.

## Features

- **Stochastic Tree Search:** Implements a novel tree search algorithm for sample generation.
- **Importance Weighting:** Automatically computes weights for each sample to improve estimation accuracy.
- **Customizable Parameters:** Easily configurable sampling parameters such as number of samples and maximum depth.
- **High Performance:** Leveraging Julia’s speed for efficient large-scale simulations.

## Installation

To install **TreeImportanceSampling.jl**, open the Julia REPL and run:

```using Pkg
Pkg.add("https://github.com/shubhg1996/TreeImportanceSampling.jl.git")
```


Alternatively, add the package to your project by including it in your `Project.toml`.

## Usage

Below is a simple example demonstrating how to use the package:

```
using TreeImportanceSampling
```

### Configure the sampling parameters

```
config = TreeSamplingConfig(
num_samples = 1000,
max_depth = 5
# additional parameters can be set as needed
)
```

### Generate samples along with their importance weights

```
samples, weights = generateImportanceSamples(config)

println("Generated $(length(samples)) samples with associated weights.")
```


This example initializes a tree sampling configuration, generates importance weighted samples, and prints the total sample count. For more advanced usage and configuration options, please refer to the package documentation or the API reference in the `docs` folder.

## Example Applications

- **Rare Event Simulation:** Estimate probabilities in high-dimensional systems where rare events are critical.
- **Monte Carlo Methods:** Enhance traditional Monte Carlo simulations with importance sampling techniques.
- **Probability Density Estimation:** Use generated samples to better approximate complex distributions.

## Contributing

Contributions are welcome! If you have ideas for improvements, bug fixes, or new features:
- Please open an issue first to discuss your ideas.
- Submit a pull request outlining your changes.

## License

**TreeImportanceSampling.jl** is released under the MIT License.

## Acknowledgements

Thank you to all contributors for helping improve this package. Special thanks to the Julia community for their continuous support and valuable feedback.
