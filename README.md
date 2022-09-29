# DCASE bioacoustics 2022
The official repository for the source code of the method presented in the paper:

    "Few-shot bioacoustic event detection using an event-length adapted ensemble of prototypical neural networks".

The method came in third place among the team submissions in the [few-shot bioacoustic event detection task](https://dcase.community/challenge2022/task-few-shot-bioacoustic-event-detection-results) during the DCASE 2022 challenge.

Please consider citing our work if the source code is helpful in your research.

# Reproduce important results of the paper
A description on how to reproduce rasults from the paper will be added soon.

## Download the data

## Pre-process the data 

Resample.

    cp -r Development_Set Development_Set_8000Hz
    for i in Development_Set_8000Hz/*/*/*.wav; do sox $i -r 8000 tmp.wav; mv tmp.wav $i; done
    
## Download pre-trained model
TODO

## Evaluate pre-trained model
TODO

## Produce plots
TODO

## Train models
Assuming you have access to three GPU:s, a simple way to train the ensemble would be:

    CUDA_VISIBLE_DEVICES=0 python main.py pcen_speech
    CUDA_VISIBLE_DEVICES=1 python main.py pcen_biodiversity
    CUDA_VISIBLE_DEVICES=2 python main.py decibel
    
the models should be in the 'experiments/final_ensemble' directory.
