from checks.check_readme_examples import *

def detect_face(type):
    if type == "Augmented":
        implement_augmented()

    elif type == "HaarCascade":
        implement_haarcascade()

    elif type == "MCNN":
        implement_mcnn()

    elif type == "RetinaFace":
        implement_retinaface()