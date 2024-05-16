import cv2
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from os import listdir
from os.path import isfile, join
from matplotlib.colors import ListedColormap

from scripts.drawers import drawAxis

def getWeightMultiplier(num: int, maxWeightMultiplier: float = 3.0, minWeightMultiplier: float = 1.0, 
                        minNumberSamples: int = 3, participants: int = 50):
    """ Returns the weight to multiply.
    """
    return ((num - minNumberSamples) / (participants - minNumberSamples)) * (maxWeightMultiplier - minWeightMultiplier) + minWeightMultiplier

class benchmarkDataset():
    """A class representing a benchmark dataset.

    This class is used to load and process a benchmark dataset for symmetry testing.
    It provides methods to access individual instances, calculate evaluation metrics,
    and visualize predictions.

    Args:
        `img_dir` (str): The directory path of the images.

        ``bench_dir`` (str): The directory path of the benchmark labels.
        
        ``pred_dir`` (str): The directory path of the predictions.

        ``test`` (str, optional): The type of test to perform. Defaults to 'all':
                *'art': Only perform the test on art images.
                *'nat': Only perform the test on natural images.
                *'all': Perform the test on all images.

        ``segmentThickness`` (int, optional): The thickness of the segments. Defaults to 10.
    """
    def __init__(self, img_dir: str, bench_dir: str, pred_dir: str, test: str = 'all', segmentThickness: int = 10) -> None:
        # Invalid test string
        if test not in ['all', 'art', 'nat', 'natural']:
            raise Exception("Invalid argument test value: {}".format(test))
            
        artCSVFiles = [join('art',f) for f in listdir(join(bench_dir, 'art')) if isfile(join(join(bench_dir, 'art'), f))]
        natCSVFiles = [join('nat',f) for f in listdir(join(bench_dir, 'nat')) if isfile(join(join(bench_dir, 'nat'), f))]

        # Art first, Natural later
        if test == 'all':
            self.label_csvs = artCSVFiles+natCSVFiles
        elif test == 'art':
            self.label_csvs = artCSVFiles
        elif test == 'nat' or test== 'natural':
            self.label_csvs = natCSVFiles
            
        # Invalid prediction size
        predictions = [f for f in listdir(pred_dir) if isfile(join(pred_dir, f))]
        if len(predictions) < len(self.label_csvs):
            raise Exception(f'Invalid number of predictions {len(predictions)} needed minimum {len(self.label_csvs)}')
        
        self.bench_dir = bench_dir
        self.img_dir = img_dir
        self.pred_dir = pred_dir
        self.segmentThickness = segmentThickness

    def __len__(self) -> int:
        return len(self.label_csvs)
    
    def __getitem__(self, index: int) -> dict:
        csvFile = self.label_csvs[index]
        imgFile = 'images_'+csvFile[:-3]+'jpg'
        predFile = csvFile[4:-3]+'txt'

        # Opening image
        image = cv2.cvtColor(cv2.imread(join(self.img_dir, imgFile)), cv2.COLOR_BGR2RGB)

        # Opening prediction
        prediction = np.loadtxt(join(self.pred_dir, predFile))

        # Opening benchmark labels
        df = pd.read_csv(join(self.bench_dir, csvFile))
        labelList = []
        bboxList = []
        for _, row in df.iterrows():
            bbox = {
                'centerX': row['centerX'],
                'centerX': row['centerX'],
                'width_box': row['width_box'],
                'height_box': row['height_box'],
                'rotation': row['rotation'],
                'num_labels': row['num_labels'],
            }
            bboxList.append(bbox)
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            drawAxis(mask, row, thickness = self.segmentThickness, color = row['num_labels'])
            labelList.append(mask)

        return {
            'image':            image,
            'masks':            labelList,
            'prediction':       prediction,
            'boundingBoxes':    bboxList
        }
    
    def getFileName(self, index) -> str:
        """ Returns the name of the file.
        """
        csvFile = self.label_csvs[index]
        return csvFile[4:-3]+'jpg'

    def changeTest(self, test: str) -> None:
        """ Changes the test type.
        """
        # Invalid test string
        if test not in ['all', 'art', 'nat', 'natural']:
            raise Exception("Invalid argument test value: {}".format(test))
            
        artCSVFiles = [join('art',f) for f in listdir(join(self.bench_dir, 'art')) if isfile(join(join(self.bench_dir, 'art'), f))]
        natCSVFiles = [join('nat',f) for f in listdir(join(self.bench_dir, 'nat')) if isfile(join(join(self.bench_dir, 'nat'), f))]

        # Art first, Natural later
        if test == 'all':
            self.label_csvs = artCSVFiles+natCSVFiles
        elif test == 'art':
            self.label_csvs = artCSVFiles
        elif test == 'nat' or test== 'natural':
            self.label_csvs = natCSVFiles

    def testOnly(self, fileNames: list[str]) -> None:
        """ Test only the indicated files.
        """
        fileNames = [line[:line.rfind('.')] + '.csv' for line in fileNames]

        temp = []
        for csvFile in self.label_csvs:
            if csvFile[4:] in fileNames:
                temp.append(csvFile)

        self.label_csvs = temp
        
    def getFalsePositives(self, index: int) -> np.ndarray: 
        """ Returns an array with the false positives of the prediction.
        """
        instance = self[index]
        falsePositives = instance['prediction'].copy()

        for i in range(len(instance['masks'])):
            non_zero = (instance['masks'][i] != 0) & (instance['prediction'] != 0)
            falsePositives[non_zero] = 0

        return falsePositives
    
    def getFalseNegatives(self, index: int) -> list[np.ndarray]: 
        """ Returns a list with arrays containing the false negatives of the prediction.
        """
        instance = self[index]
        falseNegatives = []

        for i in range(len(instance['masks'])):
            maskOne = instance['masks'][i].copy()
            maskOne[maskOne != 0] = 1
            falseNegative = maskOne - instance['prediction']
            falseNegative[falseNegative < 0] = 0
            falseNegatives.append(falseNegative)
        
        return falseNegatives
    
    def getTruePositives(self, index: int) -> list[np.ndarray]:
        """ Returns a list with arrays containing the true positives of the prediction.
        """
        instance = self[index]
        truePositives = []

        for i in range(len(instance['masks'])):
            non_zero = (instance['masks'][i] != 0) & (instance['prediction'] != 0)
            truePositive = instance['prediction'].copy()
            truePositive[~non_zero] = 0
            truePositives.append(truePositive)

        return truePositives

    def visualizePrediction(self, index: int, threshold: float = 0.5, plotWithImage: bool = True, plotJustImage = False) -> None:
        """ Plots the prediction indicating:
                * True Negatives
                * False Positives
                * False Negatives
                * True Positives
        """
        visualizationsTruePositives = []
        visualizationsFalseNegatives = []

        for i in range(len(self.getFalseNegatives(index))):
            truePositive = self.getTruePositives(index)[i].copy()
            truePositive[truePositive >= threshold] = 1
            truePositive[truePositive < threshold] = 0

            falseNegative = self.getFalseNegatives(index)[i].copy()
            falseNegative[falseNegative > (1 - threshold)] = 1
            falseNegative[falseNegative < (1 - threshold)] = 0

            visualizationsTruePositives.append(truePositive)
            visualizationsFalseNegatives.append(falseNegative)
        
        # False Positives
        visualizeFalsePositives = np.where(self.getFalsePositives(index) != 0, 1, 0).astype(int)

        # False Negatives
        visualizeFalseNegatives = np.sum(visualizationsFalseNegatives, axis=0)
        visualizeFalseNegatives = np.where(visualizeFalseNegatives != 0, 2, 0).astype(int)

        # True Positives
        visualizeTruePositives = np.sum(visualizationsTruePositives, axis=0)
        visualizeTruePositives = np.where(visualizeTruePositives != 0, 3, 0).astype(int)

        # Visualization of the prediction (no image)
        visualize = np.sum([visualizeFalsePositives, visualizeFalseNegatives, visualizeTruePositives], axis=0)

        # Visualization of the prediction (with image)
        fpBoolean = [element != 0 for element in visualizeFalsePositives]
        fnBoolean = [element != 0 for element in visualizeFalseNegatives]
        tpBoolean = [element != 0 for element in visualizeTruePositives]

        img = self[index]['image'].copy()

        img[fpBoolean] = [255,0,0]
        img[fnBoolean] = [255,255,0]
        img[tpBoolean] = [0,255,0]

        # Color legend
        label_map = {0: 'true negatives', 1: 'false positives', 2: 'false negatives', 3: 'true positives'}
        colors = ['indigo', (1,1,0), 'red', (0,1.0,0)]
        cmap = ListedColormap(colors)

        # Plotting
        if plotJustImage:
            plt.imshow(img)
            plt.axis('off')
            plt.show()
            return
        if plotWithImage:
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))

            axs[0].imshow(img)
            axs[1].imshow(visualize, cmap=cmap, interpolation='nearest')

            legend_handles = [plt.Rectangle((0,0),1,1, color=cmap(i)) for i in range(len(colors))]
            plt.legend(legend_handles, [label_map[i] for i in range(len(colors))], loc='upper right')

            plt.show()
        else:
            plt.imshow(visualize, cmap=cmap, interpolation='nearest')
            plt.title('Visualize')

            legend_handles = [plt.Rectangle((0,0),1,1, color=cmap(i)) for i in range(len(colors))]
            plt.legend(legend_handles, [label_map[i] for i in range(len(colors))], loc='upper right')

            plt.show()

    def precisionOfInstance(self, index: int, adjusted: bool = True) -> float:
        """Returns the precission of the indicated instance. Attribute adjusted is used to indicate if a 
        multiplier adjustment is applied to the calculation.
        """

        truePositivesSum = 0
        falsePositivesSum = np.sum(self.getFalsePositives(index))
        
        truePositives = self.getTruePositives(index)
        instance = self[index]
        for idx, mask in enumerate(truePositives):
            if adjusted:
                multiplier = getWeightMultiplier(max(np.unique(instance['masks'][idx])))
            else:
                multiplier = 1

            truePositivesSum += np.sum(mask*multiplier)

        if truePositivesSum + falsePositivesSum == 0:
            return 0

        precision = truePositivesSum / (truePositivesSum + falsePositivesSum)

        return precision
    
    def recallOfInstance(self, index: int, adjusted: bool = True) -> float:
        """Returns the recall of the indicated instance. Attribute adjusted is used to indicate if a 
        multiplier adjustment is applied to the calculation.
        """
        truePositivesSum = 0
        falseNegativesSum = 0

        truePositives = self.getTruePositives(index)
        falseNegatives = self.getFalseNegatives(index)

        instance = self[index]
        for idx, mask in enumerate(truePositives):
            if adjusted:
                multiplier = getWeightMultiplier(max(np.unique(instance['masks'][idx])))
            else:
                multiplier = 1

            truePositivesSum += np.sum(mask*multiplier)
            falseNegativesSum += np.sum(falseNegatives[idx]*multiplier)

        if truePositivesSum + falseNegativesSum == 0:
            return 0
        
        recall = truePositivesSum / (truePositivesSum + falseNegativesSum)

        return recall
    
    def f1OfInstance(self, index: int, adjusted: bool = True) -> float:
        """Returns the f1 score of the indicated instance. Attribute adjusted is used to indicate if a 
        multiplier adjustment is applied to the calculation.
        """
        
        precision = self.precisionOfInstance(index, adjusted=adjusted)
        recall = self.recallOfInstance(index, adjusted=adjusted)

        if precision+recall != 0:
            return (2*precision*recall)/(precision+recall)
        else:
            return 0
        
    def getPrecision(self, adjusted: bool = True) -> float:
        """Returns the precision of the dataset. Attribute adjusted is used to indicate if a 
        multiplier adjustment is applied to the calculation.
        """
        precisionSum = 0
        for i in tqdm(range(len(self))):
            precisionSum += self.precisionOfInstance(i, adjusted=adjusted)
        
        return precisionSum/len(self)

    def getRecall(self, adjusted: bool = True) -> float:
        """Returns the recall of the dataset. Attribute adjusted is used to indicate if a 
        multiplier adjustment is applied to the calculation.
        """
        recallSum = 0
        for i in tqdm(range(len(self))):
            recallSum += self.recallOfInstance(i, adjusted=adjusted)
        
        return recallSum/len(self)

    def getWF1(self, adjusted: bool = True) -> float:
        """Returns the f1 score of the dataset. Attribute adjusted is used to indicate if a 
        multiplier adjustment is applied to the calculation.
        """
        f1Sum = 0
        for i in tqdm(range(len(self))):
            added = self.f1OfInstance(i, adjusted=adjusted)
            if math.isnan(added):
                continue
            f1Sum += added
        
        return f1Sum/len(self)
    
    def benchmark(self, adjusted: bool = True) -> list[float]:
        """Returns the f1 score of the dataset for each test type (all, art, nat). Attribute adjusted is used to indicate if a
        multiplier adjustment is applied to the calculation. Type of test left as 'all' after the calculation.
        """
        tests = ['all', 'art', 'nat']
        results = {}

        for test in tests:
            print(f"Testing {test} images...")
            self.changeTest(test)
            results[test] = self.getF1(adjusted=adjusted)
        
        self.changeTest('all')

        return results     
