# The Metric is the Massage in Neural Symbolic Regression
## Comparing generalizing symbolic regression methods across metrics from the literature

## Quick setup

```
git clone git@github.com:riveSunder/metric_massage.git
cd metric_massage

virtualenv symr --python=python3.8
source symr/bin/activate

# try and run yuca tests
python -m symr_tests.test_all
```

## Benchmarks

### Generalizing Methods

This repository is designed for testing and comparing symbolic regression methods that benefit from training experience. These usually fall under the category of neural symbolic regression that uses transformers to produce mathematical expressions at inference time.

#### Neural Symbolic Regression that Scales (NSRTS) - Biggio _et al._ 2021

* [https://github.com/SymposiumOrganization/NeuralSymbolicRegressionThatScales](https://github.com/SymposiumOrganization/NeuralSymbolicRegressionThatScales)
* [arXiv:2106.06427](https://arxiv.org/abs/2106.06427)
* Biggio, Luca, et al. "Neural symbolic regression that scales." International Conference on Machine Learning. PMLR, 2021. [link](https://proceedings.mlr.press/v139/biggio21a.html)

**coming soon**

#### SymbolicGPT - Valipour _et al._ 2021

* [https://github.com/mojivalipour/symbolicgpt](https://github.com/mojivalipour/symbolicgpt)
* [arXiv:2106.14131](https://arxiv.org/abs/2106.14131)

**coming soon**

#### Symformer - Vastl _et al._ 2022

* [https://github.com/vastlik/symformer/](https://github.com/vastlik/symformer/)
* [arXiv:2205.15764](https://arxiv.org/abs/2205.15764)

**coming soon**

#### End-to-End Symbolic Regression with Transformers - Kamienny _et al._ 2022 

* [arXiv:2204.10532](https://arxiv.org/abs/2204.10532)

**coming soon (if they ever publish their code)**

### Fit Methods

While this repo is primarily designed to test symbolic regression methods that generalize, symbolic regression is in large part still dominated by methods that _fit_ each dataset from scratch.  

### PySR - Cranmer _et al._ 2019

* [https://github.com/MilesCranmer/pysr](https://github.com/MilesCranmer/pysr)
* Cite Cranmer, Miles. "PySR: Fast & parallelized symbolic regression in Python/Julia." (2020). [https://doi.org/10.5281/zenodo.4041459](https://doi.org/10.5281/zenodo.4041459)

**coming soon**
