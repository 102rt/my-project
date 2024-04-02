""" Licensed under a 3-clause BSD style license - see LICENSE.rst

This is the setup file for your allsky camera. Right now, it is setup in such a
way that it can be used with the example image data provided with
cloudynight.

(c) 2020, Michael Mommert (mommermiscience@gmail.com)
"""

import os
import logging
from scipy.stats import uniform, randint  # 统计函数模块，包含了多种概率分布的随机变量；uniform方法将随机生成下一个在指定范围内的实数

from astropy.visualization import ZScaleInterval

class ConfExample():
    """Allsky camera configuration class."""

    def __init__(self):

        # directory structure setup (note that data actually sits in
        # subdirectories that are, e.g., separated by nights)

        # define base directory
        self.DIR_BASE = os.path.join(
            ' ', *os.path.abspath(__file__).split('/')[:-2]).strip()
        # location of module base (for example data)

        # raw data directory root
        self.DIR_RAW = os.path.join(self.DIR_BASE, 'example_data')

        # archive root (will contain thumbnail images for webapp)
        self.DIR_ARCHIVE = os.path.join(self.DIR_BASE, 'workbench')

        # data directory location on host machine (where to pull FITS files from)
        self.CAMHOST_NAME = ''
        self.CAMHOST_BASEDIR = ''

        # FITS file prefix and suffix of allsky images
        self.FITS_PREFIX = ''
        self.FITS_SUFFIX = 'cr2img.fits' # fits.bz2

        # SEP parameters
        self.SEP_SIGMA = 1.5       # sigma threshold
        self.SEP_MAXFLAG = 7       # maximum source flag considered
        self.SEP_MINAREA = 5      # minimum number of pixels per source
        self.SEP_DEBLENDN = 32     # number of deblending steps 去混步骤数
        self.SEP_DEBLENDV = 0.005  # deblending parameter  去混参数
        self.SEP_BKGBOXSIZE = 64   # edge size of box for deriving background 用于导出背景的框的边缘大小 15
        self.SEP_BKGXRANGE = 3     # number of background boxes in x  x 中的背景框数
        self.SEP_BKGYRANGE = 3     # number of background boxes in y   y 中的背景框数 5

        # max solar elevation for processing (deg) 最大太阳高度（度）
       # self.MAX_SOLAR_ELEVATION = -6

        # image crop ranges (ROI must be square)
        self.X_CROPRANGE = (148,1288)  # 148,1288  700,1000
        self.Y_CROPRANGE = (135,1276) # 135,1276 690,990

        # define subregion properties
        self.N_RINGS =4
        self.N_RINGSEGMENTS =9

        # define thumbnail properties
        self.THUMBNAIL_WIDTH = 4 # inch
        self.THUMBNAIL_HEIGHT = 4 # inch
        self.THUMBNAIL_DPI = 150  # 150
        self.THUMBNAIL_SCALE = ZScaleInterval

        # mask file
        self.MASK_FILENAME = os.path.abspath(os.path.join(self.DIR_RAW,'mask.fits'))


    def update_directories(self, night):
        """prepare directory structure for a given night, provided as string
           in the form "%Y%m%d"""

        # make sure base directories exist
        os.mkdir(self.DIR_RAW) if not os.path.exists(self.DIR_RAW) else None
        os.mkdir(self.DIR_ARCHIVE) if not os.path.exists(self.DIR_ARCHIVE) else None
        self.DIR_RAW = os.path.join(self.DIR_RAW, night)
        os.mkdir(self.DIR_RAW) if not os.path.exists(self.DIR_RAW) else None

        self.DIR_ARCHIVE = os.path.join(self.DIR_ARCHIVE, night)
        os.mkdir(self.DIR_ARCHIVE) if not os.path.exists(self.DIR_ARCHIVE) else None

        self.setupLogger(night)

    def setupLogger(self, night=''):  # night=''
        # setup logging
        logging.basicConfig(
            filename=os.path.join(self.DIR_ARCHIVE, night)+'.log',
            level=logging.INFO,
            format='[%(asctime)s]: %(message)s',
            datefmt='%m/%d/%Y %H:%M:%S')
        self.logger=logging.getLogger(__name__)

        
conf = ConfExample()




from .cloudynight import AllskyImage, AllskyCamera,  ServerError    # LightGBMModel,

__all__ = ['conf', 'AllskyImage', 'AllskyCamera',
          'ServerError'] # 'LightGBMModel',    # __all__ 列表变量，存储的是当前模块中指定的一些成员（变量、函数或者类）的名称。公开接口


def scripts():
    return None


def scripts():
    return None