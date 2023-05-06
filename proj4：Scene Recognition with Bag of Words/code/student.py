import numpy as np
import matplotlib
from skimage.io import imread
from PIL import Image
from skimage import io
# from skimage.color import rgb2grey
from skimage.io import imshow
from skimage.feature import hog
from skimage.transform import resize
from numpy.linalg import norm
from scipy.spatial.distance import cdist
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import LinearSVC
from scipy.stats import mode


def get_tiny_images(image_paths):
    '''
    Inputs:
        image_paths: a 1-D Python list of strings. Each string is a complete
                     path to an image on the filesystem.
    Outputs:
        An n x d numpy array where n is the number of images and d is the
        length of the tiny image representation vector. e.g. if the images
        are resized to 16x16, then d is 16 * 16 = 256.
    '''

    #TODO: Implement this function!
    N = len(image_paths)
    size = 16
    tiny_images = []
    for i in range(N):
        image = imread(image_paths[i])
        image = resize(image, (16, 16), anti_aliasing=True).flatten()
        image = image/norm(image)
        # image = (image-np.mean(image))/np.std(image)
        tiny_images.append(image)
    tiny_images = np.asarray(tiny_images)

    return tiny_images

def build_vocabulary(image_paths, vocab_size):
    '''
    Inputs:
        image_paths: a Python list of image path strings
        vocab_size: an integer indicating the number of words desired for the
                     bag of words vocab set
    Outputs:
        a vocab_size x (z*z*9) (see below) array which contains the cluster
        centers that result from the K Means clustering.
    '''

    #TODO: Implement this function!
    cells_per_block = (4,4)
    pixels_per_cell = (4,4)
    z = cells_per_block[0]
    image_list = [imread(file) for file in image_paths]
    feature_vectors_images = []
    for image in image_list:
        feature_vectors = hog(image,feature_vector=True,cells_per_block=cells_per_block,pixels_per_cell=pixels_per_cell,visualize=False)
        feature_vectors = feature_vectors.reshape(-1,z*z*9)
        feature_vectors_images.append(feature_vectors)
    all_feature_vectors = np.vstack(feature_vectors_images)
    kmeans = MiniBatchKMeans(n_clusters=vocab_size, max_iter=2000).fit(all_feature_vectors) # change max_iter for lower compute time
    vocabulary = np.vstack(kmeans.cluster_centers_)
    
    return vocabulary

def get_bags_of_words(image_paths):
    '''
    This function should take in a list of image paths and calculate a bag of
    words histogram for each image, then return those histograms in an array.

    Inputs:
        image_paths: A Python list of strings, where each string is a complete
                     path to one image on the disk.

    Outputs:
        An nxd numpy matrix, where n is the number of images in image_paths and
        d is size of the histogram built for each image.
    '''

    vocab = np.load('vocab.npy')
    print('Loaded vocab from file.')

    #TODO: Implement this function!
    vocab_length = vocab.shape[0]
    image_list = [imread(file) for file in image_paths]
    images_histograms = np.zeros((len(image_list),vocab_length))
    cells_per_block = (4,4)
    pixels_per_cell = (4,4)
    z = cells_per_block[0]
    feature_vectors_images = []
    for i,image in enumerate(image_list):
        feature_vectors = hog(image,feature_vector=True,pixels_per_cell=pixels_per_cell,cells_per_block=cells_per_block,visualize=False)
        feature_vectors = feature_vectors.reshape(-1,z*z*9)
        histogram = np.zeros(vocab_length)
        distance = cdist(feature_vectors,vocab)
        closest_vocab = np.argsort(distance,axis=1)[:,0]
        indices, counts = np.unique(closest_vocab, return_counts=True)
        histogram[indices] += counts
        histogram = histogram / norm(histogram)
        images_histograms[i] = histogram
    return images_histograms

def svm_classify(train_image_feats, train_labels, test_image_feats):
    '''
    Inputs:
        train_image_feats: An nxd numpy array, where n is the number of training
                           examples, and d is the image descriptor vector size.
        train_labels: An nx1 Python list containing the corresponding ground
                      truth labels for the training data.
        test_image_feats: An mxd numpy array, where m is the number of test
                          images and d is the image descriptor vector size.

    Outputs:
        An mx1 numpy array of strings, where each string is the predicted label
        for the corresponding image in test_image_feats
    '''

    # TODO: Implement this function!
    clf = LinearSVC(random_state=0, tol=1e-5)
    clf.fit(train_image_feats, train_labels)
    test_predictions = clf.predict(test_image_feats)

    return test_predictions

def nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats):
    '''
    Inputs:
        train_image_feats: An nxd numpy array, where n is the number of training
                           examples, and d is the image descriptor vector size.
        train_labels: An nx1 Python list containing the corresponding ground
                      truth labels for the training data.
        test_image_feats: An mxd numpy array, where m is the number of test
                          images and d is the image descriptor vector size.
    Outputs:
        An mx1 numpy list of strings, where each string is the predicted label
        for the corresponding image in test_image_feats
    '''

    k = 9

    # Gets the distance between each test image feature and each train image feature
    # e.g., cdist
    distances = cdist(test_image_feats, train_image_feats, 'euclidean')

    #TODO:
    # 1) Find the k closest features to each test image feature in euclidean space
    # 2) Determine the labels of those k features
    # 3) Pick the most common label from the k
    # 4) Store that label in a list
    sorted_indices = np.argsort(distances, axis=1)
    knns = sorted_indices[:,0:k]
    labels = np.zeros_like(knns)
    get_labels = lambda t: train_labels[t]
    vlabels = np.vectorize(get_labels)
    labels = vlabels(knns) 
    labels = mode(labels,axis=1)[0]

    return labels
