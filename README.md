# Growing Dust in the Machine
This is the capstone project, completed in partial fulfillment of the requirements for the MS in Data Science at the University of Virginia, for André Zazzera, Kevin Hoffman, and Jae Yoon Sung. This work was done under the mentorship of Jonathan Kropko, and in collaboration with Jonathan Ramsey and Ilse Cleeves of the UVA Astronomy Department.

Within [our repository](https://github.com/andrezazz/dust-in-the-machine), you can find our various Jupyter notebooks by means of which we did our work.

Our paper can be found on arXiv [here](https://arxiv.org/abs/2104.12845).

Here is a description of the notebooks, to get you started. Happy dusting!
## Notebooks
- data_prep:
  - Dust Data Prep All Bins:
    - Goes through the simulation data models, samples random snapshots (as well as the first and last snapshots), and produces a training set.
    - Bins are normalized and negative values removed.
- mixture_density_networks:
  - Dust Classifier and MDN Combo: 
    - Test set version of Dust Classification Dists At End.
    - Loads pretrained models and predicts on test data for the 2 classifiers and 2 MDNs.
  - Dust Classification Dists At End:
    -  Classifier for whether the dust is congregated at the largest bin sizes or not. 
    -  Classifier for where the 0 point of the distribution point is. 
    -  Finally it fits 2 MDNs, one where all the dust is at the end and one where it is not.
  - MDN and Classification Training V2:
    - Improved training schema for mixture density networks, trying to approximate each dust distribution with a theoretical random distribution.
- random_forest:
  - Random Forest Exploring RMSE:
    - Error analysis on our random forest regressor, trying to see if we can identify regions of parameter space which all had bad predictions (we did not).
  - Random Forest MDN Comparison:
    - Comparing the work we did in the fall semester (MDNs) with what we did in the spring (random forest regression) to see which approach fit our problem better.
  - Random Forest Model Comparison:
    - Comparing performances of random forest models, tuning.
  - Random Forest Peak Error:
    - Comparing the distance between real and predicted modes of dust densities with Jensen-Shannon Divergence for each predicted model.
  - Random Forest Predict Bad Predictions:
    - An attempt to predict which predictions from the random forest would be inaccurate compared to the ground truth brute-force model.
  - Random Forest Run Models:
    - Demonstration of how to use some of our work, superseded by demo notebook found in package documentation [here](https://kehoffman3.github.io/astrodust/docs/demo.html).
  - Random Forest Test Models:
    - Provides code to visualize model outputs, for analysis by eye.
  - Random Forest v2 data:
    - Old attempt to use the classification ensemble approach we had needed with the MDNs, this is not reflective of our final product.
- Visualize Dust Distributions:
  - Creates graphs of output distributions from the training set, produced from a brute-force numerical approximator.
- Dust Sim Timing:
  - Timing data for the brute-force approximator, for comparison.