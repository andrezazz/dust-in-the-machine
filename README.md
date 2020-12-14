# dust-in-the-machine

## Notebooks
- 100k params:
- Dust Classification Dists At End:
  -  Classifier for whether the dust is congregated at the largest bin sizes or not. 
  -  Classifier for where the 0 point of the distribution point is. 
  -  Finally it fits 2 MDNs, one where all the dust is at the end and one where it is not.
- Dist ClassicationDists At End Params: 
  - Similar to notebook above exepct the bins have been parameterized so the MDN is predicting on 3 params and the bins are produced from the PDF
- Dust Classifier and MDN Combo: 
  - Test set version of Dust Classification Dists At End
  - Loads pretrained models and predicts on test data for the 2 classifiers and 2 MDNS
- Dust Data Prep All Bins:
  - Goes through the V1 data models, samples random snapshots and the first and last one, and produces a training set
  - Bins are normalized and negative values removed
- Dust Data Prep Regression:
  - Creates a training set for linear regression where y is the scalar mean of the output bins
- Dust MDN Training:
- Dust Ordered Logit:
  - Tries multionmial and ordered logit regression by collapsing 151 bins into 15 bins and predicts which collapsed bin has the most dust
  - Then extracts class probabilities and compares the distribution of probabilities to the output distribution
- Generate Distribution Labels:
  - Similar code to dust data prep all bins, but rather than generating an output of all the bins, fits a genextreme and sets the output to the 3 parameters describing the pdf
- Visualize Dust Distributions
  - Creates graphs of output distributions from the training set