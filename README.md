# fuseMedicalSAM

Official repo for *"Enhancing medical image segmentation by fusing pre-trained foundation models"*, 2025

## Description

This repo implements fuseSAM, a method adapted from FuseLLM to perform model fusion of multiple fine-tuned Segment Anything Models (SAMs). 

## Getting Started

### Installing

* First, clone the repo:
```
git clone 'https://github.com/nawazia/fuseMedicalSAM.git'
cd fuseMedicalSAM
git checkout gcs
```
* Next, create a conda env and setup:
```
conda create -n "fuseSAM" python=3.10
conda activate fuseSAM
cd MedSAM
pip install -e .
cd ..
```
## Usage
### Fusion

Fusing MedSAM, SAM4Med, SAM-Med2D:
```
python fuseSAM.py --target "SAM-Med2D" --data_path "data/19K/SAMed2Dv1" --json_path "data/SAMMed2D-19K.json" --device cuda --num_workers 8 --fusion i --epochs 10
```
This performs fusion into a fused 'target' model, using data from 'data_path' and 'json_path'. Fused saved model is saved to cd.

## Authors

[Ibrahim Nawaz](mailto:ibrahim.nawaz22@imperial.ac.uk)

## License

This project is licensed under the [MIT License](LICENSE)

## Acknowledgments

* [MedSAM](https://github.com/bowang-lab/MedSAM)
* [SAM4Med](https://github.com/yuhoo0302/Segment-Anything-Model-for-Medical-Images)
* [SAM-Med2D](https://github.com/OpenGVLab/SAM-Med2D)
