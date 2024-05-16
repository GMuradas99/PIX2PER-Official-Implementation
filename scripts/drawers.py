import cv2
import numpy as np

def transformKeypoints(keypoints: list, rotationMatrix: np.ndarray):
    """Returns a list with all the transformmed keypoints
    """
    result = []
    for keypoint in keypoints:
        rotatedPoint = rotationMatrix.dot(np.array(keypoint + (1,)))
        result.append((rotatedPoint[0],rotatedPoint[1]))

    return result

def getBBPoints(row, integers: bool = False):
    """Returns the rotated points for the bounding box of the indicated row
    """
    # Points for bounding box
    pts = [(row['centerX']-row['width_box']/2 , row['centerY']-row['height_box']/2), (row['centerX']+row['width_box']/2 , row['centerY']-row['height_box']/2), 
           (row['centerX']+row['width_box']/2 , row['centerY']+row['height_box']/2), (row['centerX']-row['width_box']/2 , row['centerY']+row['height_box']/2)]
    
    # Rotating box
    center = (row['centerX'], row['centerY'])
    rotation_matrix = cv2.getRotationMatrix2D(center, -row['rotation'], 1)
    pts = transformKeypoints(pts, rotation_matrix)

    # Return integers if needed
    if integers:
        intPoints = []
        for pt in pts: 
            intPoints.append([round(pt[0]), round(pt[1])])
        return intPoints

    return pts

def drawAxis(mask, row, thickness = 3, color = 1):
    """Draws symmetry axis of desired thickness and color on the mask
    """
    pts = getBBPoints(row)

    if 'axis' in row:
        orientation = row['axis']
    else:
        orientation = "vertical_axis"
    
    # Drawing symmetry axis
    if orientation == "vertical_axis":
        startAxis = ((pts[0][0] + pts[1][0])/2  , (pts[0][1] + pts[1][1])/2)
        endAxis = ((pts[2][0] + pts[3][0])/2  , (pts[2][1] + pts[3][1])/2) 
    elif orientation == "horizontal_axis":
        startAxis = ((pts[0][0] + pts[3][0])/2  , (pts[0][1] + pts[3][1])/2)
        endAxis = ((pts[1][0] + pts[2][0])/2  , (pts[1][1] + pts[2][1])/2)    

    cv2.line(mask, (int(startAxis[0]),int(startAxis[1])), (int(endAxis[0]),int(endAxis[1])), color, thickness)

    return mask