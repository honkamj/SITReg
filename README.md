# SITReg: Multi-resolution architecture for symmetric, inverse consistent, and topology preserving image registration

Official implementation of SITReg, a deep learning intra-modality image registration arhitecture fulfilling strict symmetry properties.

## Winner of Learn2reg 2024 LUMIR task

The architecture was the backbone of the winning submission for Learn2reg 2024 LUMIR task on unsupervised brain MRI registration (https://learn2reg.grand-challenge.org/learn2reg-2024/). See the instructions for reproducing the results below.

![Learn2reg 2024 LUMIR metrics for SITReg](learn2reg_results.png "Learn2reg 2024 LUMIR results")

## Environment setup

First install conda (https://docs.conda.io/en/latest/). To setup the enviroment navigate to directory ''devenv'' and execute ''setup.py'' with Python 3:

    python setup.py

The setup script will create a virtual enviroment with required packages installed. To activate the environent navigate to root and run:

    conda activate ./.venv

The repository has been designed to be used with Visual Studio Code (https://code.visualstudio.com/).

## Usage

For this section we assume that you have navigated to directory ''src'' and have activated the environment.

### Training

Here you can find commands which can be used to train the model with different datasets. If you run into problems with reproducing the results in the paper, do not hesitate to contact the authors.

The scripts will download the datasets to DATA_ROOT_PATH and the models will be saved to TRAINING_ROOT_PATH inside the directory MODEL_NAME. Note that the automatic data downloading will not work if using multiple devices (in those cases, just run the command with single "--devices cpu" flag for data downloading first), or if the data is no longer available at the url specified within the code (and Lung250M-4B dataset is only downloaded partially).

For all the trainings which were part of the paper we chose the best epoch based on metrics computed on validation set (see [Evaluation](#evaluation)). Trainings are somewhat heavy but converge fast, and some configs might have unneccesarily large number of epochs. E.g. Lung250M-4B training converges already during the first epoch (5000 training pairs) and no improvement is seen after that.

**OASIS:**

To train similar model used in the paper with affinely aligned brain OASIS dataset (https://www.oasis-brains.org/) from Learn2Reg (https://learn2reg.grand-challenge.org/), run the following command:

    python train.py --config scripts/configs/sitreg/oasis/cc_grad_1.0.json --training-root TRAINING_ROOT_PATH --data-root DATA_ROOT_PATH --num-workers 4 --model-name MODEL_NAME --devices cuda:0

To train similar model used in the paper with raw (not affinely aligned) brain OASIS dataset (https://www.oasis-brains.org/) from Learn2Reg (https://learn2reg.grand-challenge.org/), run the following command:

    python train.py --config scripts/configs/sitreg/oasis/cc_grad_1.0_raw_data.json --training-root TRAINING_ROOT_PATH --data-root DATA_ROOT_PATH --num-workers 4 --model-name MODEL_NAME --devices cuda:0

**LPBA40:**

To train similar model used in the paper with LPBA40 dataset (https://resource.loni.usc.edu/resources/atlases-downloads/), run the following command:

    python train.py --config scripts/configs/sitreg/lpba40/cc_grad_1.0_very_very_deep.json --training-root TRAINING_ROOT_PATH --data-root DATA_ROOT_PATH --num-workers 4 --model-name MODEL_NAME --devices cuda:0

**Lung250M-4B:**

To train similar model used in the paper with Lung250M-4B dataset (https://github.com/multimodallearning/Lung250M-4B), run the following command:

    python train.py --config scripts/configs/sitreg/lung250m_4b/cc_grad_1.0_very_deep_half_res.json --training-root TRAINING_ROOT_PATH --data-root DATA_ROOT_PATH --num-workers 4 --model-name MODEL_NAME --devices cuda:0

Note that the Lung250M-4B config is not identical to the one used in the paper, as it includes masking out invalid regions for similarity loss, which improves the results. The feature was disabled in the experiments since other methods did not have such property, and we wanted to compare architectures, not loss functions. One can replicate the results in the paper by setting "ignore_mask" to "True" in the config. Also, we do not use the keypoints provided as part of the training set.

**LUMIR (Learn2reg 2024 winner):**

To train similar model to our submission to Learn2reg 2024 LUMIR task, first run the following command:

    torchrun --standalone --nnodes=1 --nproc-per-node=4 train.py --config scripts/configs/sitreg/lumir/cc_grad_1.0_very_deep_heavy.json --training-root TRAINING_ROOT_PATH --data-root DATA_ROOT_PATH --num-workers 4 --model-name MODEL_NAME --devices cuda:0 --devices cuda:1 --devices cuda:2 --devices cuda:3

After the first training has finnished, run the following fine-tuning training with group consistency loss and NDV loss (use the same values for TRAINING_ROOT_PATH, DATA_ROOT_PATH, and MODEL_NAME as it will continue the previous training):

    torchrun --standalone --nnodes=1 --nproc-per-node=3 train.py --config scripts/configs/sitreg/lumir/cc_grad_1.0_very_deep_heavy_group_consistency_ndv.json --training-root TRAINING_ROOT_PATH --data-root DATA_ROOT_PATH --num-workers 4 --model-name MODEL_NAME --devices cuda:0 --devices cuda:1 --devices cuda:2

The trainings are designed for 4 and 3 gpus respectively, and we recommend to use high-end GPU such as A100 or H100. The first training can easily be modified for smaller number of GPUs by reducing the batch size in the config and modifying the training command. The latter training is only implemented for multiples of 3 gpus (due to the group consistency loss), and is very memory hungry (would not fit easily on a single gpu).

**NLST:**

To train model for NLST from Learn2Reg (https://learn2reg.grand-challenge.org/), run the following command:

    python train.py --config scripts/configs/sitreg/nlst/cc_grad_1.0_very_deep_heavy.json --training-root TRAINING_ROOT_PATH --data-root DATA_ROOT_PATH --num-workers 4 --model-name MODEL_NAME --devices cuda:0

For this dataset no proper hyperparameter tuning has been done.

### Evaluation

To evaluate a model, run the command

    python -m scripts.inference --training-root TRAINING_ROOT --data-root DATA_ROOT --division DIVISION --model-name MODEL_NAME --devices cuda:0 --evaluate --do-not-save-outputs --epoch EPOCHS

DIVISION should be either "validate" or "test". EPOCHS defines the epochs for which you want to run the evaluation.

### Inference

To register images and store the results, run the command

    python -m scripts.inference --training-root TRAINING_ROOT --data-root DATA_ROOT --division DIVISION --model-name MODEL_NAME --devices cuda:0 --epoch EPOCHS

DIVISION should be either "validate" or "test". EPOCHS defines the epochs for which you want to run the inference.

## Deformation inversion layer

If you are only interested in *deformation inversion layer*, a neural network component for inverting deformation fields, see repository [deformation inversion layer](https://github.com/honkamj/deformation-inversion-layer "deformation inversion layer").

## Composable mapping

As part of the project a PyTorch utility library for handling coordinate mappings (deformations) between images (2D or 3D) was developed, see repository [composable mapping](https://github.com/honkamj/composable-mapping "composable mapping").

## Publication

If you use the repository, please cite (see [bibtex](citations.bib)):

- **SITReg: Multi-resolution architecture for symmetric, inverse consistent, and topology preserving image registration**  
[Joel Honkamaa](https://github.com/honkamj "Joel Honkamaa"), Pekka Marttinen  
The Journal of Machine Learning for Biomedical Imaging (MELBA) ([10.59275/j.melba.2024-276b](https://doi.org/10.59275/j.melba.2024-276b "10.59275/j.melba.2024-276b"))

## Acknowledgments

Small parts of the repository are rewritten from [NITorch](https://github.com/balbasty/nitorch), [VoxelMorph](https://github.com/voxelmorph/voxelmorph), [TorchIR](https://github.com/BDdeVos/TorchIR), [DeepReg](https://github.com/DeepRegNet/DeepReg), and [SciPy](https://scipy.org/).

Also, [tutorial](http://implicit-layers-tutorial.org/) by Zico Kolter, David Duvenaud, and Matt Johnson was very helpful in implementing the deformation inversion layer.

## License

SITReg is released under the MIT license.
