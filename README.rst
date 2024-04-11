.. These are examples of badges you might want to add to your README:
   please update the URLs accordingly

    .. image:: https://api.cirrus-ci.com/github/<USER>/nps-tdes.svg?branch=main
        :alt: Built Status
        :target: https://cirrus-ci.com/github/<USER>/nps-tdes
    .. image:: https://readthedocs.org/projects/nps-tdes/badge/?version=latest
        :alt: ReadTheDocs
        :target: https://nps-tdes.readthedocs.io/en/stable/
    .. image:: https://img.shields.io/coveralls/github/<USER>/nps-tdes/main.svg
        :alt: Coveralls
        :target: https://coveralls.io/r/<USER>/nps-tdes
    .. image:: https://img.shields.io/pypi/v/nps-tdes.svg
        :alt: PyPI-Server
        :target: https://pypi.org/project/nps-tdes/
    .. image:: https://img.shields.io/conda/vn/conda-forge/nps-tdes.svg
        :alt: Conda-Forge
        :target: https://anaconda.org/conda-forge/nps-tdes
    .. image:: https://pepy.tech/badge/nps-tdes/month
        :alt: Monthly Downloads
        :target: https://pepy.tech/project/nps-tdes
    .. image:: https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter
        :alt: Twitter
        :target: https://twitter.com/nps-tdes

|

Neural Processes for Tidal Disruption Events
============================================

This is a project to develop a neural process model for the classification and prediction 
of tidal disruption events from identified transients. This project will address the 
specific needs of TDEs in order to confidently disambiguate them from imposter phenomena 
(i.e. active galactic nuclei, supernovae, etc) that are much better represented in 
current classification software. 

Current neural network classification systems are limited by the lack of a unifying 
physical model for the expected behavior of TDEs and are unable to adapt to new 
observations once a network has been trained. NPs use a neural network to parameterize 
and learn a map from the observed data to posterior predictive distributions. Like GPs, 
this will allow for rapid adaptation to new data while providing uncertainty estimation 
using the nature of probabilistic inference. Unlike GPs, however, it will not require 
strictly establishing priors, which are instead learned from the data.

Goals
-----

- Create a model for predicting light curve behavior given progressively 
  increasing context points.
- Use this predicted light curve to classify the transient, giving a confidence with
  which the object may be considered a TDE.
- Comparatively evaluate the efficacy of the model on simulated and real data, and
  anticipate its usefulness in the era of the Vera Rubin Observatory (LSST).

Useful Links
------------

- [PLAsTiCC Data](https://zenodo.org/records/2539456): archive of simulated light curve
  data for multiple bands, meant to mimic the output photometry of LSST.

