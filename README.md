# statistical-signal-comparison
This repository provides tools for the statistical classification of signals by analyzing the difference between the Fourier Transforms (FFTs) of signal pairs. The classification is based on whether the two signals are identical, differing only due to noise, or if they are distinct signals. The differences in their FFTs exhibit distinct properties that can be used to conduct a statistical test to determine the similarity or difference between the signals.

For detailed information on the project, refer to the report.pdf.

### Repository structure
##### -`main_classification_pipeline`
This folder includes all the functions and scripts required to generate a dataset, prepare it for analysis by computing the pairwise differences between the FFTs of the signals, and perform a Shapiro-Wilks test to assess whether these differences follow a normal distribution. A normal distribution of the differences suggests that the two signals are identical, differing only by the noise.

##### -`supplementary material`
This folder contains the functions required to implement all the alternative approaches to data preparation and analysis as described in the report.
