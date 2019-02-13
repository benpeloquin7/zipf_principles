<!-- 

# The interactions of rational, pragmatic agents lead to efficient language structure and use

Code and paper for Cogsci 2019.

## Abstract

Languages display a diverse set of distributional regularities such as the relation between a word's frequency and rank in a corpus, the distribution of dependency lengths, or the presence of lexical properties such as ambiguity. We discuss a framework for studying how these properties emerge from in-the-moment interactions of rational, pragmatic speakers and listeners. Our work takes Zipfian notions of lexicon-level efficiency as a starting point, connecting these ideas to Gricean notions of conversational-level efficiency. To do so, we derive an objective function for measuring the communicative efficiency of linguistic systems and then examining the behavior of this objective in a series of simulations focusing on the communicative function of ambiguity in language. These simulations suggest that rational pragmatic agents will produce communicatively efficient systems and that interactions between such agents provide a framework for examining efficient properties of language more broadly.

## Simulation 1: Optimal languages contain ambiguity when context is informative

![Context effects](/paper/figs/fig1.png?raw=true "Ambiguity is increasingly efficient as a function of amount of context.")


### Generate simulation data

To generate the simulated data for this section run from repo root.
```
>>> python -m ambiguity.run --sim-type context --out-dir your_output_dir
```
or the simulation data from the paper is available upon request (bpeloqui@stanford.edu)

### Generate plots

To generate the plots run code in `ambiguity/simulation-runners/context-ambiguity-plots.Rmd` pointing to local file-paths.


## Simulation 2: Rational, pragmatic speakers use ambiguity efficiently

![Context effects](/paper/figs/discourse_grid_plot.png?raw=true "Rational pragmatic agents use ambiguity efficiently over discourse.")

### Generate simulation data

To generate the simulated data run code chunks in `ambiguity/simulation-runners/discourse-ambiguity.Rmd` or the simulation data from the paper is available upon request.

### Generate plots

To generate the plots run code in `ambiguity/simulation-runners/discourse-ambiguity-plots.Rmd` pointing to local file-paths.

## Supplemental materials

We provide a complete derivation of the Zipfian-inspired objective for language efficiency as well as additional modeling details in the supplemental materials `paper/supplementary_materials.pdf`.

## Repo organization:

`zipf_principles/ambiguity/...`

* `run.py` primary run script for Simulation 1.
* `config.py` contains configurations used for Simulation 1.
* `objectives.py` contains various objectives including our derived speaker-listener cross-entropy as well as several comparisons not included in the paper.
* `simulations.py` primary simulation infrastructure. Note that we run "context" simulations in the current work.
* `agents.py` basic matrix manipulation as well as matrix-defined RSA agents.
* `utils.py` general usefulities.
* `simulation-runners` contains simulation run and plotting functionality (detailed above).
* `zipf_principles/ambiguity/webppl/...`
  * `discourse_ambiguity_model.wppl` contains Simulation 2 model code. For best viewing set interpreter to javascript.

 -->