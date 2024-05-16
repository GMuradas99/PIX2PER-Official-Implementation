## From Pixels to Perception: A Benchmark for Human-like Symmetry Detection

<p align="center">
Gonzalo Muradas Odriozola, Lisa Koßmann, Tinne Tuytelaars, Johan Wagemans
</p>

Official Python implementation of *From Pixels to Perception: A Benchmark for Human-like Symmetry Detection*.

### Mathematical explanation

The PIX2PER uses a novel WF1 metric for measuring the acuracy of reflection symmetry detection. The next equations describe how WF1 is calculated.

$$
Weight Multiplier(NumInstancesInCluster) = \frac{Num Instances In Cluster - Min Samples Per Cluster}{Num Participants - Min Samples Per Cluster} * (MaxMultiplier-1) + 1
$$

$$
WeightedPrecision = \frac{\sum_{i=1}^{n}True Positives_i*Weight Multiplier(NumInstancesInCluster_i)}{\sum_{i=1}^{n} [True Positives_i * Weight Multiplier(NumInstancesInCluster_i)] + False Positives}
$$

$$
Weighted Recall = \frac{\sum_{i=1}^{n}True Positives_i*Weight Multiplier(NumInstancesInCluster_i)}{\sum_{i=1}^{n}(True Positives_i + False Negatives_i)*Weight Multiplier(NumInstancesInCluster_i)}
$$

$$
WF1 = \frac{2 * WeightedPrecision * Weighted Recall}{WeightedPrecision+Weighted Recall}
$$

### Usage

Store predictions in arrays with the same size as the original images from PIX2PER, normalized from 0.0 to 1.0 in .txt files. 

```
    python benchmark.py -pd PATH/TO/PREDICTION/DIRECTORY
```
