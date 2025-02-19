import scipy.constants as scic


def get_light_speed():
    return scic.c


def get_nano():
    return scic.nano


def get_micro():
    return scic.micro


def get_milli():
    return scic.milli


def get_centi():
    return scic.centi


def get_giga():
    return scic.giga


def get_planck_constant():
    return scic.h


def get_elementary_charge():
    return scic.e


def convert_nm_to_m(thicknesses):
    return thicknesses * get_nano()


def convert_m_to_nm(thicknesses):
    return thicknesses / get_nano()


def convert_m_to_um(thicknesses):
    return thicknesses / get_micro()


def convert_mm_to_m(thicknesses):
    return thicknesses * get_milli()


def convert_cm_to_m(thicknesses):
    return thicknesses * get_centi()
