from recognition.Face_Recogniton_Model import mtcnn as mt
from recognition.Face_Recogniton_Model import inception_resnet_v1
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F
from torchvision import datasets
from recognition.folder import ImageFolder
import numpy as np
from PIL import Image
import time

def collate_fn(x):
    return x[0]

def face_distance(face_encodings, face_to_compare):
    """
    Given a list of face encodings, compare them to a known face encoding and get a euclidean distance
    for each comparison face. The distance tells you how similar the faces are.

    :param face_encodings: List of face encodings to compare
    :param face_to_compare: A face encoding to compare against
    :return: A numpy ndarray with the distance for each face in the same order as the 'faces' array
    """

    if len(face_encodings) == 0:
        return np.empty((0))
    # finding the Euclidean Distance
    return np.linalg.norm(face_encodings - face_to_compare)


def compare_faces(known_face_encodings, face_encoding_to_check, tolerance=0.75):
    """
    Compare a list of face encodings against a Known Faces to see if they match.

    :param known_face_encodings: A list of known face encodings
    :param face_encoding_to_check: A single Unknown face encoding to compare against the Known Face list
    :param tolerance: How much tolerance distance between faces to consider it a match. Lower is more strict. 0.6 is typical best performance.
    :return: A list of True/False values indicating which known_face_encodings match the face encoding to check
    """
    # Finding the distances between the each known face and unknown face by using --> face_distance function.
    distances = np.array([face_distance(e1, face_encoding_to_check) for e1 in known_face_encodings])
    print(distances)
    # If the distance is below Tolerance value then its converted to TRUE else FALSE

    # Returns list of TRUE or FALSE for each known image.
    return list(distances <= tolerance)


def find_match(known_faces, names, face,tol=0.75):
    """
        Compare a list of face encodings against a candidate encoding to see if they match.

        :param known_faces: A list of known face encodings
        :param names: A single face encoding to compare against the list
        :param face: A single face encoding to compare against the list
        :param tol: How much tolerance distance between faces to consider it a match. Lower is more strict. 0.6 is typical best performance.
        :return: A list of True/False values indicating which known_face_encodings match the face encoding to check
        """
    # just replacing "_" with " " from each person name.
    corrected_names = []
    for name in names:
        corrected_name = name.replace("_", " ").strip()
        corrected_names.append(corrected_name)

    # Comparing the Known Faces with the Unknown Face passed in Parameter
    matches = compare_faces(known_faces, face,tol)

    # count variable is used to get the index of matched face
    count = 0
    for match in matches:
        if match:
            # if the unknown face matches with any known face then it returns the Name of Person it got recognised with.
            return corrected_names[count]
        count += 1
    return 'Access Denied'


def face_detection_and_recognition(unknown_image):


    # MTCNN (FACE DETECTION MODEL) INITIALISATION
    # user can change the image_size parameter to their desired size.
    # Optimal values will range from 128 to 512
    # higher size means more time taken by Models to process everything
    device = torch.device('cpu')
    mtcnn = mt.MTCNN(image_size=256,
                     margin=20,
                     min_face_size=50,
                     thresholds=[0.33, 0.6, 0.7],
                     factor=0.709,
                     select_largest=True,
                     post_process=True,
                     device=device)

    # INCEPTION RESNET (FACE RECOGNITION MODEL) INITIALISATION
    # Using pre-trained model (VGGFACE2) which is placed inside checkpoints folder
    resnet = inception_resnet_v1.InceptionResnetV1(pretrained='vggface2').eval().to(device)

    # Defining the location of Known Faces Path to look
    # This folder will contain all known people faces user want to recognise
    dataset = ImageFolder('static/known_faces/')

    # All the names are assigned to index.
    dataset.idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}

    # Loading the Dataset Folder using DataLoader Function
    loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=4)

    aligned = []
    names = []

    # Looping through each known Face image and detecting the face
    # If face is found then its appended to aligned variable else left out.
    for x, y in loader:

        # Detecting the face in each image (x) and returning aligned face as output.
        x_aligned, prob = mtcnn(x, return_prob=True)
        if x_aligned is not None:
            aligned.append(x_aligned)
            names.append(dataset.idx_to_class[y])

    # Concatenating tensors
    aligned = torch.stack(aligned).to(device)

    # Embedding the stacked image encodings
    embeddings = resnet(aligned).detach().cpu()

    # Converting the Image format from any other format like (CMYK)to RGB
    # unknown_image is Pillow Image Type
    unknown_image = unknown_image.convert('RGB')

    # Passing the unknown image to detect face using MTCNN Model
    aligned_unknown, prob = mtcnn(np.array(unknown_image), return_prob=True)

    # In case the image don't contain any face then it returns "Face Not Detected" as output
    if prob == None:
        return "Face Not Detected"

    # Concatenating the Unknown Image to have same dimensions as known images Data set had
    aligned_unknown = torch.stack([aligned_unknown]).to(device)

    # Embedding the unknown Person Face to be used to match with known Faces
    embeddings_unknown = resnet(aligned_unknown).detach().cpu()[0]

    # Finding the first match found from known Faces
    match = find_match(embeddings, names, embeddings_unknown)

    # Returns the Person Name with whom unknown Face matches
    return match