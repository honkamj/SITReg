# SITReg: Multi-resolution architecture for symmetric, inverse consistent, and topology preserving image registration using deformation inversion layers

SITReg is a deep learning intra-modality image registration arhitecture fulfilling strict symmetry properties.

## Environment setup

First install conda (https://docs.conda.io/en/latest/). To setup the enviroment navigate to directory ''devenv'' and execute ''setup.py'' with Python 3:

    python setup.py

The setup script will create a virtual enviroment with required packages installed. To activate the environent navigate to root and run:

    conda activate ./.venv

The repository has been designed to be used with Visual Studio Code (https://code.visualstudio.com/) and it's Python extension (https://marketplace.visualstudio.com/items?itemName=ms-python.python).

## Usage

For this section we assume that you have navigated to directory ''src'' and have activated the environment.

### Training

To train the model used in the paper with affinely aligned brain OASIS dataset (https://www.oasis-brains.org/) from Learn2Reg (https://learn2reg.grand-challenge.org/), run the following command:

    python -m scripts.train --config scripts/configs/sitreg/oasis/cc_grad_1.0_small_disp.json --training-root TRAINING_ROOT_PATH --data-root DATA_ROOT_PATH --num-workers 4 --model-name MODEL_NAME --devices cuda:0

To train the model used in the paper with raw (not affinely aligned) brain OASIS dataset (https://www.oasis-brains.org/) from Learn2Reg (https://learn2reg.grand-challenge.org/), run the following command:

    python -m scripts.train --config scripts/configs/sitreg/oasis/cc_grad_1.0_small_disp_raw_data.json --training-root TRAINING_ROOT_PATH --data-root DATA_ROOT_PATH --num-workers 4 --model-name MODEL_NAME --devices cuda:0

To train the model used in the paper with LPBA40 dataset (https://resource.loni.usc.edu/resources/atlases-downloads/), run the following command:

    python -m scripts.train --config scripts/configs/sitreg/lpba40/cc_grad_1.0_small_disp_very_deep.json --training-root TRAINING_ROOT_PATH --data-root DATA_ROOT_PATH --num-workers 4 --model-name MODEL_NAME --devices cuda:0

The scripts will download the datasets to DATA_ROOT_PATH and the models will be saved to TRAINING_ROOT_PATH inside the directory MODEL_NAME. Note that the automatic data downloading will not work if using multiple devices or if the data is no longer available at the url specified within the code.

### Evaluation

To evaluate a model, run the command

    python -m scripts.inference --training-root TRAINING_ROOT --data-root DATA_ROOT --division DIVISION --model-name MODEL_NAME --devices cuda:0 --evaluate --do-not-save-outputs --epoch EPOCHS

DIVISION should be either "validate" or "test". EPOCHS defines the epochs for which you want to run the evaluation.

### Inference

To register images and store the results, run the command

    python -m scripts.inference --training-root TRAINING_ROOT --data-root DATA_ROOT --division DIVISION --model-name MODEL_NAME --devices cuda:0 --epoch EPOCHS

DIVISION should be either "validate" or "test". EPOCHS defines the epochs for which you want to run the inference.

## Deformation inversion layer

If you are only interested in *deformation inversion layer*, a neural network component for inverting deformation fields, see repository [deformation inversion layer](https://github.com/honkamj/deformation_inversion_layer "deformation inversion layer").

## Publication

If you use the repository, please cite (see [bibtex](citations.bib)):

- **SITReg: Multi-resolution architecture for symmetric, inverse consistent, and topology preserving image registration using deformation inversion layers**  
[Joel Honkamaa](https://github.com/honkamj "Joel Honkamaa"), Pekka Marttinen  
Under review ([preprint arXiv:2303.10211](https://arxiv.org/abs/2303.10211 "preprint arXiv:2303.10211"))

## Acknowledgments

Small parts of the repository are rewritten from [NITorch](https://github.com/balbasty/nitorch), [VoxelMorph](https://github.com/voxelmorph/voxelmorph), [TorchIR](https://github.com/BDdeVos/TorchIR), [DeepReg](https://github.com/DeepRegNet/DeepReg), and [SciPy](https://scipy.org/).

## License

SITReg is released under the MIT license.
