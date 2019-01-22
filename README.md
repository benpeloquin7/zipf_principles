# Pragmatic interactions lead to efficient language structure and use

Code and paper for cogsci 2019.

### Abstract

We discuss a framework for studying how the distributional properties of linguistic systems emerge from in-the-moment interactions of speakers and listeners. Our work takes Zipfian notions of lexicon-level efficiency as a starting point, connecting these ideas to Gricean notions of conversational-level efficiency. To do so, we begin by deriving an objective function for measuring the communicative efficiency of linguistic systems and then examining the behavior of this objective in a series of simulations focusing on the communicative function of ambiguity in language. These simulations suggest that rational pragmatic agents will produce communicatively efficient systems.

## Efficient language design

To generate the simulated data for this section run from repo root.
```
>>> python -m ambiguity.run --sim-type context --out-dir your_output_dir
```
or you may access the simulation data from the paper [here](https://web.stanford.edu/~bpeloqui/Projects/zipf_principles/data/).

#### Files:

`zipf_principles/ambiguity/...`

* `run.py` primary run script.
* `config.py` contains the simulation configuration used in the paper.
* `objectives.py` contains various objectives including our derived speaker-listener cross-entropy as well as several comparisons not included in the paper.
* `simulations.py` primary simulation infrastructure. Note that we run "context" simulations in the current work.
* `agents.py` basic matrix manipulation as well as matrix-defined RSA agents.
* `utils.py` general usefulities.

## Efficient language use

Code for discourse simulations is written in [webppl](http://webppl.org/).

#### Files:

`zipf_principles/ambiguity/webppl/...`

* `discourse_ambiguity_model.wppl` contains model code. For best viewing set interpreter to javascript.
* `notebooks/discourse-ambiguity.Rmd` contains simulation run and plotting code.
