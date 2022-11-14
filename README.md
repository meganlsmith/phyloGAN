# phyloGAN

## Introduction

phyloGAN is a Generative Adversarial Network (GAN) that infers phylognetic relationships. phyloGAN takes as input a concatenated alignments, or a set of gene alignments, and then infers a phylogenetic tree either considering or ignoring gene tree heterogeneity.

## Installation

### Depedencies 

phyloGAN requires several python packages, along with AliSim.

#### Python packages
*Handling trees and alignments:* Biopython, dendropy, ete3

*Machine learning:* tensorflow

*Miscellaneous utilities:* copy, datetime, io, itertools, matplotlib, numpy, os, random, re, scipy, sys

#### AliSim
phyloGAN was developed using the version of AliSim distributed with IQ-TREE v2.2.0 (Beta). A version of IQ-TREE with AliSim must be installed, and the user must provide the path to the executable.

### phyloGAN
To install phyloGAN, clone the GitHub repository:

    git clone github.com/meganlsmith/phyloGAN

## phyloGAN (concatenation version)

The original version of phyloGAN takes as input a concatenated alignment and infers a phylogenetic tree. 

### Input Files

#### Concatenated alignment

The concatenated alignment should be provided in phylip format. For an example see `test_data/concatenated_test.phy`.

#### Parameters file

The only other input to phyloGAN is the parameters file. For an example see `test_data/params_concatenated.txt`. In the example file, each line is described in a comment (following '#'). A few things to note:

* The temporary folder provided will be deleted during the run. *DO NOT* use an existing folder. This must be a new directory. If using an HPC, scratch spaces may be ideal because a lot of I/O to this directory will occur.
* The pseudoobserved setting is only recommended to be used in specific development contexts. Datasets are simulated from branch lengths drawn from an exponential distribution with mean lambda, and this is likely not desired in most simulation studies.
* It is recommended that users begin with a 'Random' start tree, rather than a 'NJ' start tree. Beginning with the 'NJ' tree seems to cause issues because the generated data seen early in training is fairly good.




