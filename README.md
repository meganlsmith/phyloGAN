# phyloGAN

## Introduction

phyloGAN is a Generative Adversarial Network (GAN) that infers phylognetic relationships. phyloGAN takes as input a concatenated alignments, or a set of gene alignments, and then infers a phylogenetic tree either considering or ignoring gene tree heterogeneity.

## Installation

### Depedencies 

phyloGAN requires several python packages, along with AliSim.

#### Python packages
##### Handling trees and alignments
Biopython, dendropy, ete3
##### Machine learning
tensorflow
##### Miscellaneous utilities
copy, datetime, io, itertools, matplotlib, numpy, os, random, re, scipy, sys

#### AliSim
phyloGAN was developed using the version of AliSim distributed with IQ-TREE v2.2.0 (Beta). A version of IQ-TREE with AliSim must be installed, and the user must provide the path to the executable.

### phyloGAN
To install phyloGAN, clone the GitHub repository:
    git clone github.com/meganlsmith/phyloGAN.

## phyloGAN (concatenation version)

The original version of phyloGAN takes as input a concatenated alignment and infers a phylogenetic tree. 

### Input Files
