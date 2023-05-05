# InterFair 



## Summary

<!-- start summary -->


*InterFair* is our Bias Detection Tool Entry for the [#ExpeditionHacks competition on bias in healthcare](https://expeditionhacks.com/bias-detection-healthcare/). 
Our entry uses a new fair machine learning framework called **F**airness **Oriented** Multiobjective **O**ptimization, or [Fomo](https://cavalab.org/fomo). 
Interfair allows any interested healthcare entity (a hospital system, insurance payor, or individual clinic) to feed in an ML model for a given prediction task, measure its performance across intersectional groups of patients, and optimize it with respect to several flexible fairness constraints.


## Overview

In this repository we provide two scripts that measure bias (`measure_disparity.py`) and correct for it (`mitigate_disparity.py`). 
We also provide a demonstration that uses these scripts to measure and mitigate bias in models that predict risk of emergency department admission. 
Our demonstration is based on the recently released [MIMIC-IV electronic health record dataset](https://www.nature.com/articles/s41597-022-01899-x). 

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

See the [Demo: Measuring Disparity](https://github.com/cavalab/interfair/blob/main/docs/demo_measure_disparity.ipynb) for additional info. 

### Mitigating Model Disparity

To use `mitigate_disparity.py`, you must first have a [pandas](https://pandas.pydata.org/) DataFrame containing observations, the true labels, and a variable number of demographic columns. 
You can then run the following from the command line:

```python
python mitigate_disparity.py --dataset your_dataset.csv
```

See the [Demo: Mitigating Disparity](https://github.com/cavalab/interfair/blob/main/docs/demo_mitigate_disparity.ipynb) for additional info. 

<!-- end basic -->

## License

<!-- start license -->

Interfair is licensed under BSD 3.  See [LICENSE](https://github.com/cavalab/fomo/blob/main/LICENSE).

<!-- end license -->

## Contact 

<!-- start contact -->

Team: [Willam La Cava](https://williamlacava.com) and [Elle Lett](https://ellelett.com)

Team Lead Contact: williamlacava@gmail.com

When they are not competing in hackathons, William and Elle can be found conducting research at the [Cava lab](https://cavalab.org), part of the [Computational Health Informatics Program](https://chip.org) at Boston Children's Hospital.

<!-- end contact -->
