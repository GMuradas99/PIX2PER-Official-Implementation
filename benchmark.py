import os

import pandas as pd

from tqdm import tqdm

from scripts.config import get_parser
from WF1 import benchmarkDataset

if __name__ == '__main__':
    args = get_parser()
    
    pix2per = benchmarkDataset(img_dir=os.path.join(args.dataset_dir, 'images'),
                               bench_dir=os.path.join(args.dataset_dir, 'labels'),
                               pred_dir=args.prediction_dir,
                               segmentThickness=args.segment_thickness)

    fileNames = []
    weightedPrecissions = []
    weightedRecalls = []
    weightedF1s = []

    for i in tqdm(range(len(pix2per))):
        fileNames.append(pix2per.getFileName(i))
        weightedPrecissions.append(pix2per.precisionOfInstance(i))
        weightedRecalls.append(pix2per.recallOfInstance(i))
        weightedF1s.append(pix2per.f1OfInstance(i))

    fileNames.append('AVERAGE')
    weightedPrecissions.append(sum(weightedPrecissions) / len(weightedPrecissions))
    weightedRecalls.append(sum(weightedRecalls) / len(weightedRecalls))
    weightedF1s.append(sum(weightedF1s) / len(weightedF1s))

    df = pd.DataFrame({'fileName': fileNames,
                       'weighted precission': weightedPrecissions,
                       'weighted recall': weightedRecalls,
                       'weighted r1': weightedF1s})

    df.to_csv(args.output_dir, index=False)

    print(f'Average WF1:{weightedF1s[-1]:.2f}')