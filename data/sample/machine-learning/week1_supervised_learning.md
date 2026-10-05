# Week 1: Supervised Learning

## Definitions

Supervised learning trains a model on labelled examples: pairs of inputs and the
correct outputs. Regression predicts a continuous value (house price), while
classification predicts a discrete class (spam or not spam).

## Bias-variance tradeoff

A model with high bias underfits: it is too simple to capture the pattern, so
error is high on both training and test data. A model with high variance overfits:
it memorises noise in the training set and generalises poorly. Increasing model
complexity lowers bias but raises variance; the goal is the sweet spot that
minimises total expected error.

## Regularisation

Regularisation adds a penalty on weight size to the loss function. L2 (ridge)
shrinks all weights smoothly toward zero. L1 (lasso) drives some weights exactly
to zero, which performs feature selection. The strength is controlled by the
hyperparameter lambda, chosen with cross-validation.

## Cross-validation

In k-fold cross-validation the data is split into k folds; the model is trained
k times, each time holding out one fold for validation, and the scores are
averaged. It gives a more reliable estimate of generalisation than a single
train/test split, especially on small datasets.
