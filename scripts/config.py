import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='PIX2PER benchmark arguments')
    parser.add_argument('-dd','--dataset_dir', default='./PIX2PER', type=str,
                        help='Path to directory containing PIX2PER\'s images and labels')
    parser.add_argument('-st','--segment_thickness', default=10, type=int,
                        help='Segment thickness in pixels')
    parser.add_argument('-pd', '--prediction_dir', type=str,
                        help='Path to directory containign predictions in .txt format')
    parser.add_argument('-o', '--output_dir', default='./results.csv', type=str,
                        help='Path to output file')

    args = parser.parse_args()

    return args