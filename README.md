# Code repository for paper "Split Conformal Prediction under Data Contamination"

This repository contains code to re-produce all the results, plots and tables presented within the paper.

# Classification
## Synthetic Data
To reproduce the results with the synthetic data, use the command

```
python3 main.py --c {config_file}
```
The names of the config files can be found in the ```experiment_configs/``` directory.

To reproduce the results presented across different types of classifiers, run
```
python3 main.py -c logistic_by_model.yaml
python3 main.py -c hypercube_by_model.yaml
```
Then use the notebook ```synthetic_by_model_table.ipynb``` to generate the table.

Similarly for the experiment varying epsilon, run
```
python3 main.py -c logistic_by_eps.yaml
```

Then use the notebook ```logistic_by_eps.ipynb``` to generate the plot.

For both the table and the plots, you will need to copy and paste the name of your run into the notebook, which can be found in the 
```results/``` directory. 

## Real Data
For the real data, you will need to begin by downloading the CIFAR10N dataset from http://noisylabels.com/. Place the files 
```CIFAR-10_human.npy``` and ```CIFAR10_human.pt``` into the ```data/CIFAR10N``` folder.

The first step is to train the ResNet18 models for each noise setting. This is done by running
```
python3 main.py train_resnets_cifar10n.py  
```
We recommend using a GPU for this, although the device is just set to ```auto``` in pytorch lightning.
This script will create a directory structured as 
```
models/
- aggre_label/
- clean_label/
- random_label1/
    ...
```
with one folder per label noise setting. Each folder will contain the trained model as well as the predicted logits for
the given split of the data.

After running this, use the command
```
python3 main.py -c cifar10n.yaml
```

## Regression
Finally, the regression plot can be re-created using the ```regression_plot.ipynb``` notebook.
