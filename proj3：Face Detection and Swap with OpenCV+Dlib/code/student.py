import cv2
import numpy as np
import dlib

class TooManyFaces(Exception):
    pass

class NoFaces(Exception):
    pass

def get_landmarks(detector, predictor, img):
    
    '''
    This function first use `detector` to localize face bbox and then use `predictor` to detect landmarks (68 points, dtype: np.array).
    
    Inputs: 
        detector: a dlib face detector
        predictor: a dlib landmark detector, require the input as face detected by detector
        img: input image
        
    Outputs:
        landmarks: 68 detected landmark points, dtype: np.array

    '''
    
    #TODO: Implement this function!
    # Your Code to detect faces
    faces = detector(img)
    
    if len(faces) > 1:
        raise TooManyFaces
    if len(faces) == 0:
        raise NoFaces
    
    # Your Code to detect landmarks
    landmarks = [[p.x,p.y] for p in predictor(img,faces[0]).parts()]
    landmarks = np.array(landmarks)

    return landmarks

def get_face_mask(img, landmarks):
    
    '''
    This function gets the face mask according to landmarks.
    
    Inputs: 
        img: input image
        landmarks: 68 detected landmark points, dtype: np.array
        
    Outputs:
        convexhull: face convexhull
        mask: face mask 

    '''
    
    #TODO: Implement this function!
    mask = np.zeros_like(img)
    convexhull = cv2.convexHull(landmarks)
    cv2.fillConvexPoly(mask,convexhull,255)

    return convexhull, mask

def get_delaunay_triangulation(landmarks, convexhull):
    
    '''
    This function gets the face mesh triangulation according to landmarks.
    
    Inputs: 
        landmarks: 68 detected landmark points, dtype: np.array
        convexhull: face convexhull
        
    Outputs:
        triangles: face triangles 
    '''
    
    #TODO: Implement this function!
    rect = cv2.boundingRect(convexhull)
    subdiv = cv2.Subdiv2D(rect)
    subdiv.insert(landmarks.tolist())
    triangles = subdiv.getTriangleList()
    triangles = np.array(triangles,dtype=np.int32)

    return triangles

def transformation_from_landmarks(source_landmarks, target_landmarks):
    '''
    Return an affine transformation [s * R | T] such that:
        sum ||s*R*p1,i + T - p2,i||^2
    is minimized.
    
    Inputs: 
        target_landmarks: 68 detected landmark points of the target face, dtype: np.array
        source_landmarks: 68 detected landmark points of the source face that need to be warped, dtype: np.array
        
    Outputs:
        triangles: face triangles 
    '''
    # Solve the procrustes problem by subtracting centroids, scaling by the
    # standard deviation, and then using the SVD to calculate the rotation. See
    # the following for more details:
    #   https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
    
    #TODO: Implement this function!
    target_landmarks = np.matrix(target_landmarks.astype(np.float64))
    source_landmarks = np.matrix(source_landmarks.astype(np.float64))
    c1 = np.mean(target_landmarks,axis=0)
    c2 = np.mean(source_landmarks,axis=0)
    target_landmarks -= c1
    source_landmarks -= c2
    s1 = np.std(target_landmarks)
    s2 = np.std(source_landmarks)
    target_landmarks /= s1
    source_landmarks /= s2
    U,S,Vt = np.linalg.svd(target_landmarks.T * source_landmarks)
    R = (U * Vt).T
    M = np.vstack([np.hstack(((s2/s1)*R,c2.T-(s2/s1)*R*c1.T)),np.matrix([0.,0.,1.])])
    
    return M

def warp_img(img, M, target_shape):
    '''
    This function utilizes the affine transformation matrix M to transform the img.
    
    Inputs: 
        img: input image (np.array) need to be warped.
        M: affine transformation matrix.
        target_shape: the image shape of target image
        
    Outputs:
        warped_img: warped image.
    
    '''
    
    #TODO: Implement this function!
    warped_img = np.zeros_like(img)
    cv2.warpAffine(img,
                   M[:2],
                   (target_shape[1],target_shape[0]),
                   dst=warped_img,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)
    
    return warped_img