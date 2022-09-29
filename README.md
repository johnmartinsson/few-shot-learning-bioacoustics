# DCASE bioacoustics 2022
The official repository for the source code of the method presented in the paper:

    "Few-shot bioacoustic event detection using an event-length adapted ensemble of prototypical neural networks".

The method came in third place among the team submissions in the [few-shot bioacoustic event detection task](https://dcase.community/challenge2022/task-few-shot-bioacoustic-event-detection-results) during the DCASE 2022 challenge.

Please consider citing our work if the source code is helpful in your research.

# Reproduce important results of the paper
A description on how to reproduce rasults from the paper will be added soon.

## Download the data

We will
- download the challenge data into the ./data directory, 
- extract the data from the zip file, 
- rename (or copy) the directory, 
- and resample all the audio data to 22050Hz __in-place__.

A suggestion of commands to do so:

    cd data
    wget https://zenodo.org/record/6482837/files/Development_Set.zip
    unzip Development_Set.zip
    mv Development_Set Development_Set_22050Hz # cp -r Development_Set Development_Set_22050Hz  # (if you want to keep a copy)
    sh resample.sh
    
## Download pre-trained model

    wget https://www.dropbox.com/s/ad7jxb8z5b06tjd/final_ensemble.zip
    unzip final_ensemble.zip

## Evaluate pre-trained model

    jupyther notebook
    
## Produce plots
TODO

## Train models
Assuming you have access to three GPU:s, a simple way to train the ensemble would be:

    CUDA_VISIBLE_DEVICES=0 python main.py pcen_speech
    CUDA_VISIBLE_DEVICES=1 python main.py pcen_biodiversity
    CUDA_VISIBLE_DEVICES=2 python main.py decibel
    
the models should be in the 'experiments/final_ensemble' directory.
