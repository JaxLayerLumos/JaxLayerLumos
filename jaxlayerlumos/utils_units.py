import scipy.constants as scic


def get_light_speed():
    return scic.c


def get_nano():
    return scic.nano


def get_milli():
    return scic.milli


def get_centi():
    return scic.centi


def get_giga():
    return scic.giga


def convert_nm_to_m(thicknesses):
    return thicknesses * get_nano()


def convert_mm_to_m(thicknesses):
    return thicknesses * get_milli()


def convert_cm_to_m(thicknesses):
    return thicknesses * get_centi()
