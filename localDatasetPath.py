def getDatasetPath(name):
    lower_case_name = name.lower()

    if lower_case_name == "alessia":
        return "C:/Users/Utente/OneDrive/Desktop/BioID-FaceDatabase-V1.2/"
    elif lower_case_name == "francesco":
        return "C:/Users/fborz/OneDrive/Documenti/ComputerVision/dataset/"
    elif lower_case_name == "carmine":
        return "C:/Users/Carmine Grimaldi/Desktop/CV Dataset/"
    else:
        return None


def getMtflDatasetPath(name):
    lower_case_name = name.lower()

    if lower_case_name == "alessia":
        return None
    elif lower_case_name == "francesco":
        return None
    elif lower_case_name == "carmine":
        return "C:/Users/Carmine Grimaldi/Desktop/CV Dataset/MTFL/"
    else:
        return None
