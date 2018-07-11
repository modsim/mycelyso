# -*- coding: utf-8 -*-
"""
The OME-TIFF reader module implements an ImageStack able to open OME-TIFF images, by using the tifffile module
to open the particular tiff file in general and custom XML processing code to parse the OME annotation in particular.
"""

from ..imagestack import Dimensions
from .tiff import TiffImageStack

from xml.etree import cElementTree as ElementTree


class OMETiffImageStack(TiffImageStack):
    extensions = ('.ome.tif', '.ome.tiff',)

    priority = 500

    def open(self, location, **kwargs):
        super(OMETiffImageStack, self).open(location, **kwargs)

        self.fp = self.tiff.pages[0]
        if not self.fp.is_ome:
            raise RuntimeError('Not an OMETiffStack')
        self.xml = None
        self.ns = ''
        self.xml_str = self.fp.description

        try:
            self.treat_z_as_mp = bool(self.parameters['treat_z_as_mp'])
        except KeyError:
            self.treat_z_as_mp = False

        self.images = self._parse_ome_xml(self.xml_str)

        an_image = next(iter(next(iter(self.images.values()))))

        self.set_dimensions_and_sizes(
            [Dimensions.Time, Dimensions.PositionXY, Dimensions.PositionZ, Dimensions.Channel],
            [an_image['SizeT'], len(self.images), an_image['SizeZ'], an_image['SizeC']])

    @staticmethod
    def pixel_attrib_sanity_check(pa):
        """

        :param pa:
        :raise RuntimeError:
        """
        if pa['BigEndian'] == 'true':
            raise RuntimeError("Unsupported Pixel format")
        if pa['Interleaved'] == 'true':
            raise RuntimeError("Unsupported Pixel format")

    def _parse_ome_xml(self, xml):
        try:  # bioformats seem to copy some (wrongly encoded) strings verbatim
            root = ElementTree.fromstring(xml)
        except ElementTree.ParseError:
            root = ElementTree.fromstring(xml.decode('iso-8859-1').encode('utf-8'))

        self.xml = root

        self.ns = ns = root.tag.split('}')[0][1:]

        # sometimes string properties creep up, but then we don't care as we don't plan on using them ...
        def float_or_int(s):
            """

            :param s:
            :return:
            """
            try:
                if '.' in s:
                    return float(s)
                else:
                    return int(s)
            except ValueError:
                return s

        keep_pa = {'SizeZ', 'SizeY', 'SizeX', 'SignificantBits', 'PhysicalSizeX', 'PhysicalSizeY', 'SizeC', 'SizeT'}

        images = {}

        if self.treat_z_as_mp:  # handling for mal-encoded files
            image_nodes = [n for n in root.getchildren() if n.tag == ElementTree.QName(ns, 'Image')]
            # there will be only one image node
            imn = image_nodes[0]

            pixels = [n for n in imn.getchildren() if n.tag == ElementTree.QName(ns, 'Pixels')][0]

            pa = pixels.attrib

            self.pixel_attrib_sanity_check(pa)

            pai = list({k: v for k, v in pa.items() if k in keep_pa}.items())

            tiff_data = {
                (n.attrib['FirstC'], n.attrib['FirstT'], n.attrib['FirstZ']): n.attrib
                for n in pixels.getchildren() if n.tag == ElementTree.QName(ns, 'TiffData')
            }
            planes = [dict(
                list(n.attrib.items()) +
                list(tiff_data[(n.attrib['TheC'], n.attrib['TheT'], n.attrib['TheZ'])].items()) + pai
            ) for n in pixels.getchildren() if n.tag == ElementTree.QName(ns, 'Plane')]

            planes = [{k: float_or_int(v) for k, v in p.items()} for p in planes]
            multipoints = range(planes[0]['SizeZ'])
            images = {mp: [p for p in planes if p['TheZ'] == mp] for mp in multipoints}
            # more fixing

            def _correct_attributes(inner_p, inner_planes):
                inner_p['PositionX'] = inner_planes[0]['PositionX']
                inner_p['PositionY'] = inner_planes[0]['PositionY']
                inner_p['PositionZ'] = inner_planes[0]['PositionZ']
                inner_p['TheZ'] = 0
                return inner_p

            images = {mp: [_correct_attributes(p, planes) for p in planes] for mp, planes in images.items()}

        else:
            image_nodes = [n for n in root.getchildren() if n.tag == ElementTree.QName(ns, 'Image')]
            for n, imn in enumerate(image_nodes):
                pixels = [n for n in imn.getchildren() if n.tag == ElementTree.QName(ns, 'Pixels')][0]

                pa = pixels.attrib

                self.pixel_attrib_sanity_check(pa)

                pai = list({k: v for k, v in pa.items() if k in keep_pa}.items())

                tiff_data = {
                    (n.attrib['FirstC'], n.attrib['FirstT'], n.attrib['FirstZ']): n.attrib
                    for n in pixels.getchildren() if n.tag == ElementTree.QName(ns, 'TiffData')
                }
                planes = [dict(
                    list(n.attrib.items()) +
                    list(tiff_data[(n.attrib['TheC'], n.attrib['TheT'], n.attrib['TheZ'])].items()) + pai
                ) for n in pixels.getchildren() if n.tag == ElementTree.QName(ns, 'Plane')]

                planes = [{k: float_or_int(v) for k, v in p.items()} for p in planes]

                images[n] = planes

        return images

    def get_data(self, what):

        try:
            channel = what[Dimensions.Channel]
        except KeyError:
            channel = 0

        tps = self.images[what[Dimensions.PositionXY]]
        tp = [tp for tp in tps if tp['TheT'] == what[Dimensions.Time] and tp['TheC'] == channel][0]

        return self.tiff.pages[tp['IFD']].asarray()

    def get_meta(self, what):

        tps = self.images[what[Dimensions.PositionXY]]
        image = [tp for tp in tps if tp['TheT'] == what[Dimensions.Time]][0]

        position = self.__class__.Position(x=image['PositionX'], y=image['PositionY'], z=image['PositionZ'],)
        meta = self.__class__.Metadata(time=image['DeltaT'], position=position, calibration=image['PhysicalSizeX'])
        return meta
