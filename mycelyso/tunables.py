# -*- coding: utf-8 -*-

"""
This file contains all the tunables available in mycelyso.

You can set them via :code:`-t Name=value` on the command line.
"""

from tunable import Tunable
from .processing import binarization as binarization_module


class NodeEndpointMergeRadius(Tunable):
    """ Radius in which endpoints are going to be merged [µm] """
    default = 0.5


class NodeJunctionMergeRadius(Tunable):
    """ Radius in which junctions are going to be merged [µm] """
    default = 0.5


class NodeLookupRadius(Tunable):
    """ Radius in which nodes will be searched for found pixel structures [µm] """
    default = 0.5


class NodeLookupCutoffRadius(Tunable):
    """ Radius at which nodes will be ignored if they are further away [µm] """
    default = 2.5


class NodeTrackingJunctionShiftRadius(Tunable):
    """ Maximum search radius for junctions [µm·h⁻¹] """
    default = 5.0


class NodeTrackingEndpointShiftRadius(Tunable):
    """ Maximum search radius for endpoints [µm·h⁻¹] """
    default = 100.0


class CropWidth(Tunable):
    """ Crop value (horizontal) of the image [pixels] """
    default = 0


class CropHeight(Tunable):
    """ Crop value (vertical) of the image [pixels] """
    default = 0


class BoxDetection(Tunable):
    """ Whether to run the rectangular microfluidic growth structure detection as ROI detection """
    default = False


class StoreImage(Tunable):
    """ Whether to store images in the resulting HDF5. This leads to a potentially much larger output file. """
    default = False


class SkipBinarization(Tunable):
    """ Whether to directly use the input image as binary mask. Use in case external binarization is desired. """
    default = False


class CleanUpGaussianSigma(Tunable):
    """Clean up step: Sigma [µm] used for Gaussian filter"""
    default = 0.075


class CleanUpGaussianThreshold(Tunable):
    """Clean up step: Threshold used after Gaussian filter (values range from 0 to 1)"""
    default = 0.5


class CleanUpHoleFillSize(Tunable):
    """Clean up step: Maximum size of holes [µm²] which will be filled"""
    default = 1.0


class RemoveSmallStructuresSize(Tunable):
    """Remove structures up to this size [µm²]"""
    default = 10.0


class BorderArtifactRemovalBorderSize(Tunable):
    """Remove structures, whose centroid lies within that distance [µm] of a border"""
    default = 10.0


class TrackingMaximumRelativeShrinkage(Tunable):
    """ Tracking, maximal relative shrinkage """
    default = 0.2


class TrackingMinimumTipElongationRate(Tunable):
    """ Tracking, minimum tip elongation rate [µm·h⁻¹]"""
    default = -0.0


class TrackingMaximumTipElongationRate(Tunable):
    """ Tracking, maximum tip elongation rate [µm·h⁻¹] """
    default = 100.0


class TrackingMaximumCoverage(Tunable):
    """ Tracking, maximum covered area ratio at which tracking is still performed """
    default = 0.2


class TrackingMinimumTrackedPointCount(Tunable):
    """ Tracking, minimal time steps in track filter [#] """
    default = 5


class TrackingMinimalMaximumLength(Tunable):
    """ Tracking, minimal hyphae end length in track filter [µm] """
    default = 10.0


class TrackingMinimalGrownLength(Tunable):
    """ Tracking, minimal hyphae gained length in track filter [µm] """
    default = 5.0


class ThresholdingTechnique(Tunable):
    """ Binarization method to use, for available methods see documentation (mycelyso.processing.binarization) """

    @classmethod
    def test(cls, value):
        return value in dir(binarization_module)

    default = "experimental_thresholding"


class ThresholdingParameters(Tunable):
    """ Parameters for the used binarization method, passed as key1:value1,key2:value2,... string """
    default = ""
