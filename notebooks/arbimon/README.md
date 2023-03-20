# Evaluate "pattern matching" in Arbimon on DCASE 2022

This is an evaluation of the pattern matching method in Arbimon on the DCASE
2022 dataset.

## Pattern Matching

TODO: is there a technical description of the "pattern matching" method?

My understanding of the pattern matching method is that it simply does per-pixel
correlation between the template and query templates in the dataset.

That is, it is a template matching method.

## Evaluation

I have manually entered the 5-shot annotations from the DCASE 2022 validation
dataset into Arbimon. Unfortunately, I found no way to exactly adjust the
templates so that they match the timings of the annotations. I have entered them
as best as I could.

In the DCASE challenge we only have timings for the 5-shot, no information on
the frequency range, but I have also added templates which I call "tight", where
I have to the best of my ability made a tight template of the sounds.

TODO: add image.

![alt text](./figures/ME1_arbimon_annotations.png)

