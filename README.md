# Incremental learning in image classification: an ablation study
Incremental learning is a learning paradigm in which a deep architecture is required to continually learn from a stream of data.

In our work, we implemented several state-of-the-art algorithms for incremental learning, such as Finetuning, [Learning Without Forgetting](https://arxiv.org/abs/1606.09282) and [iCaRL](https://arxiv.org/abs/1611.07725). Next, we amend several modifications to the origial iCaRL algorithm: specifically, we experiment with different combinations of distillation and classification losses and introduce new classifiers into the framework.\
Furthermore, we propose three extensions of the origial iCaRL algorithm and we verify their effectiveness. We perform our tests on CIFAR-100, as used in the original iCaRL paper.

A [fully detailed report](https://github.com/irenebenedetto/MLDL-incremental-learning-project/blob/master/Incremental_Learning_report.pdf) of the project is available.

## Frameworks
- PyTorch
- scikit-learn

## Proposed algorithms
- `models.py` containts a modular implementation of iCaRL and Learning without Forgetting within a single class `FrankenCaRL`
- `knn_icarl.py` is a version of iCaRL which employs a normal kNN instead of a nearest class mean classifier
- `svmCaRL.py` is a SVM-iCaRL hybrid, loosely inspired by [SupportNet](https://arxiv.org/abs/1806.02942)
- `specialist.py` is a version of iCaRL which makes use of an ensemble of specialized models for each class batch
- `FamiliCaRL.py` adopts a more convoluted "double distillation" mechanism to further reduce the imbalance between new and past classes
- `exemplars_generator.py` contains various functions that attempt to synthesize new exemplars based on the ones already stored in memory
