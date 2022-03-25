# DCASE bioacoustics 2022
A repository for the DCASE bioacoustics 2022 challenge.

# Resample data
    cp -r Development_Set Development_Set_8000Hz
    for i in Development_Set_8000Hz/*/*/*.wav; do sox %i -r 8000 tmp.wav; mv tmp.wav $i; done
