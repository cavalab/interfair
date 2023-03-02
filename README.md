# InterFair with FOMO: Intersectional Fairness with Fairness-Oriented Multiobjective Optimization (FOMO)



## Summary

<!-- start summary -->


This repository contains Interfair's Bias Detection Tool Entry for the [#ExpeditionHacks competition on bias in healthcare](https://expeditionhacks.com/bias-detection-healthcare/). 
Our entry is based on a new fair machine learning framework we developed called **F**airness **Oriented** Multiobjective **O**ptimization, or [Fomo](https://cavalab.org/fomo). 

FOMO allows any interested healthcare entity (a hospital system, insurance payor, or individual clinic) to feed in an ML model for a given prediction task and optimize it with respect to several flexible fairness constraints.


## Overview

In this repository we provide two scripts that use Fomo to measure bias (`measure_disparity.py`) and correct for it (`mitigate_disparity.py`). 
We also provide a demonstration that uses these scripts to measure and mitigate bias in models that predict risk of emergency department admission. 
Our demonstration is based on the recently released [MIMIC-IV electronic health record dataset](https://www.nature.com/articles/s41597-022-01899-x). 

### Repository structure:

- `measure_disparity.py`: 
- `mitigate_disparity.py`: 

<!-- end summary -->

## Basic Usage

<!-- start basic -->

### Installation

```text
pip install -r requirements.txt
```


### Measuring Model Disparity

To use `measure_disparity.py`, you must first create a [pandas](https://pandas.pydata.org/) DataFrame containing outputs from your model on a set of observations, the true labels, and a variable number of demographic columns. 
You can then run the following from the command line:

```python
python measure_disparity.py --dataset your_dataset.csv
```

See the [Demo: Measuring Disparity](https://cavalab.org/interfair/demo_measure_disparity.html) for additional info. 

### Mitigating Model Disparity

To use `mitigate_disparity.py`, you must first have a [pandas](https://pandas.pydata.org/) DataFrame containing observations, the true labels, and a variable number of demographic columns. 
You can then run the following from the command line:

```python
python mitigate_disparity.py --dataset your_dataset.csv
```

See the [Demo: Mitigating Disparity](https://cavalab.org/interfair/demo_measure_disparity.html) for additional info. 

<!-- end basic -->

## License

<!-- start license -->

Interfair is licensed under BSD 3.  See [LICENSE](https://github.com/cavalab/fomo/blob/main/LICENSE).

<!-- end license -->

## Contact 

<!-- start contact -->

Team: Willam La Cava and Elle Lett
Team Lead Contact: williamlacava@gmail.com

<!-- end contact -->
