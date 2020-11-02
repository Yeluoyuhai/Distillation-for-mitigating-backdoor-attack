# Distillation for mitigating backdoor attack
Distillation for mitigating backdoor attack (Implementation for AISEC 2020 paper)

## Paper

Disabling Backdoor and Identifying Poison Data by using Knowledge Distillation in Backdoor Attacks on Deep Neural Networks

Kota Yoshida, Takeshi Fujino (Ritsumeikan university)

## installation

Clone the repository and run 
```
$ pip install .
```

if you need install as development mode:
```
$ pip install -e .
```

## Reproducibility testing

Experimental code is placed in `experiments/<target dataset>/`

### Distillation

1. Run `Prepare_datasets.ipynb`
  
  Split dataset for experiment.

  You can set the random seeds in the notebook.

  Default seed is `np.random.seed(20200620)`.

2. Run `train_baseline_model.ipynb`
   
  Train a baseline model

3. Run `train_backdoor_model.ipynb`
   
  Train a backdoor model

4. Run `train_distilled_model.ipynb`
   
  Train a distilled model

5. Run `Dataset_screening.ipynb`
   
  Screen the poison training dataset and pick the negative samples

6. Run `train_fine_tuned_model.ipynb`
   
   Fine-tune a fine_tuned model with negative samples