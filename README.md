# SIGNAL
Dataset for Semantic and Inferred Grammar Neurological Analysis of Language

## Repository structure

This repository follows the next structure:
```
├── stimuli_generation                  # Linguistic stimuli preparation
|   ├──stimuli_check                    # Code for estimation and selection of stimuli parameters
|   ├──break_grammar                    # Code for generation of grammatically incongruent sentences  
|   └──break_semantics                  # Code for generation of semantically incongruent sentences                
├── EEG_processing                      # Source code for EEG data analysis
|   ├── z-scores_estimation             # Code for pairwise conditions comparison in EEG data
|   └── draw_plots                      # Code for visualisation
├── LLM_processing                      # Source code for LLM data analysis
|   └── LLM_probing                     # Code for pairwise condition comparison of LLM data
├── STIMULI.xlsx                        # Dataset with linguistic stimuli and their main parameters
├── README.md                           # README file
└── requirements.txt                    # A file with requirements 
```

## Dataset

In this paper, we present SIGNAL, a dataset for **S**emantic and **I**nferred **G**rammar **N**eurological **A**nalysis of **L**anguage. Our dataset contains 600 Russian language sentences along with the 64-channel EEG recordings from humans reading these sentences in a carefully designed experimental paradigm.

The dataset include well-controlled stimuli balanced on key lexical-semantic properties and controlled syntactic structure including sentence groups distinguished by three syntactic structures and four congruency conditions (semantical, grammatical, and semantical-grammatical).

The possible syntactic structures were:
- Subject + VERB + OBJECT
  - _Avtory poluchili podarki_
  - /Authors received presents/
- SUBJECT + VERB + ADJECTIVE + OBJECT
  - _Dramaturg pridumal sovremenniy syujet_
  - /Writer invented modern storyline/
- SUBJECT + VERB + OBJECT+ GENITIVE
  - _Programma pokajet mestopolozhenie predmeta_
  - /The programm will show location of the item/

The congruency conditions were  semantical, grammatical, or semantical-grammatical (in)congruency of the **Object** argument within each sentence:

- **Congruent sentence**: 
  - *Storony podpisali* **soglashenie**. 
  - /The parties signed an agreement (accusative)/
- **Semantically incongruent sentence**: 
  - *Storony podpisali **detstvo**.* 
  - /The parties signed childhood (accusative)/
- **Grammatically incongruent sentence**: 
  - *Storony podpisali **soglashenii***. 
  - /The parties signed an agreement (locative)/
- **Semantically and gramatically incongruent sentence**: 
  - *Storony podpisali **detstve***. 
  - /The parties signed childhood (locative)/

Anomalous stimuli were generated using language model, and validity of them was checked via an online validation study with 133 respondents to prove that (in)congruence type is correctly identified by Russian native speakers.
The reliability and interpretability of dataset was proven by EEG estimation results and LLMs probing.

## Stimuli generation

The code allows to 
- control congruent sentences for balance of lexical-semantic parameters
- generate a semantically/grammatically incongruent counterpart of the congruent sentence

To generate semantically incongruent sentences run the following script:

```bash
python break_semantics.py --input congruent_sentences.csv --output sem_inconguent.csv
```

To generate grammatically incongruent sentences run the following script:

```bash
python break_grammar.py --input congruent_sentences.csv --output gram_inconguent.csv
```

## EEG analysis

EEG data include recordings of 21 participants revealing a statistical difference between stimuli congruence conditions on a neuro-physiological level.

The code allows to:
- ```z-scores_estimation.py```
  - compute averageg event-related potential data within each condition 
  - compute z-scores to estimate pairwise differences between congruency conditions
  - compute statistically significance of the results via permutation tests
  - obtain significant spatial-temporal clusters contrasting ERP between four congruency conditions
- ```draw_plots.py```
  - visualise z-score estimation 
  - make topographical plots of significantly different clusters

The results demonstrated the presence of significant topically organized neurolinguistically plausible differences in the EEG data between incongruity conditions.

![](./EEG_processing/topoplot.png)

The preprocessed and epoched EEG data is available at https://huggingface.co/datasets/zhuravlevahana/SIGNAL/tree/main.

## LLM probing

LLM probing data include experiments for the probing validation study (including supplementary tokenization effect study) and the algorithm of layer-wise condition contrasting based on ruBERT LLM activations. LLM probing allows for model inference and subsequent diagnostic classification study on datasets compatible with the one used in the study. 

`SIGNAL_SPREADSHEET` should be replaced with the link to the spreadsheet containing the data of interest.

We applied Representational Similarity Analysis ( 30) to evaluate activation difference between 12 types of stimuli (three groups of sentences different by syntax structure each divided into four congruency conditions) detected by LLMs. As a result, we obtained layer-wise Representational Dissimilarity Matrices (RDMs) contrasting each pair of condition presented
