""" Licensed under a 3-clause BSD style license - see LICENSE.rst

cloudynight - Tools for automated cloud detection in all-sky camera data

(c) 2020, Michael Mommert (mommermiscience@gmail.com)
"""


import os  # python环境下对文件，文件夹执行操作的一个模块
import time

import requests  # 抓取网页图片
import shlex # shlex 模块最常用的是 split() 函数，用来分割字符串，通常与 subprocess 结合使用
import subprocess
import datetime
from joblib import dump, load # 属于sklearn模块，用于训练模型的保存与加载
from collections import OrderedDict # 获取一个有序的字典对象

import numpy as np
from scipy.signal import convolve2d  # 计算2维数组的卷积
# import matplotlib; matplotlib.use('Agg')  # 用来配置matplotlib的backend （后端）的命令。
# 所谓后端，就是一个渲染器，用于将前端代码渲染成我们想要的图像，Agg 渲染器是非交互式的后端，没有GUI界面，所以不显示图片，它是用来生成图像文件
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import gaussian_filter  # 高斯滤波，去除小尺度特征 scipy.ndimage.filters
import sep  # 天文源提取和测光库，用于分割和分析天文图像。它读取 FITS 格式文件，执行一系列可配置的任务，包括背景估计、源检测、去混入和各种源测量，最后输出 FITS 格式目录文件。
from astropy.io import fits # astropy基于python语言，天文数据处理中最常用的包之一
from astropy.time import Time
from astropy.visualization import ImageNormalize, LinearStretch
from scipy.stats import skew
from skimage import measure  # 基于scipy的一款图像处理包，它将图片作为numpy数组进行处理，是数字图像处理工具
from lightgbm import LGBMClassifier # lightGBM分类器，图像识别与分类
from sklearn.model_selection import (train_test_split, cross_validate,
                                     RandomizedSearchCV) # 分割测试集与训练集、交叉验证、随机搜索（寻找训练结果最优的一组参数）
from sklearn.metrics import f1_score, confusion_matrix # 验证指标 f1 score、混淆矩阵
from astroplan import Observer # astroplan:天文包

# define configuration class here
from . import conf # 相对路径导入
# from cloudynight import conf  # ConfExample
import numpy as np
from skimage import measure
from scipy.ndimage import zoom

# define observatory
observatory = None # Observer.at_site('Mustag Observatory')


class AllskyImageError(Exception):  # AllskyImageError是一个自定义的异常类，继承自Exception,报告与AllskyImage类相关的异常情况。


    pass

class ServerError(Exception):  # 异常类，同样继承自Exception。它用于表示与服务器相关的异常情况。
    pass


class AllskyImage():
    """Class for handling individual FITS image files."""
    # AllskyImage类有一个构造函数__init__，接收三个参数filename（图像文件名）、data（图像数据数组）、header（FITS头文件）。
    # 构造函数将这些参数保存到类的实例变量中。除了这些参数之外，还有一些其他实例变量，如datetime（台址日期）、thumbfilename（缩略图文件名）、subregions（子区域数组）和features（提取的特征）。

    def __init__(self, filename, data, header):
        self.filename = filename  # 图像文件名
        self.datetime = None      # 台址日期
        self.thumbfilename = None
        self.data = data          # 图像数据数组
        self.header = header      # fits头文件
        self.subregions = None    # 子区域数组
        self.features = None      # 提取的特征
        
    @classmethod
    def read_fits(cls, filename):
        # 类方法，用于从FITS图像文件创建AllskyImage实例。它接收一个参数filename（FITS图像文件名）。

        """Create `~AllskyImage` instance from FITS image file.

        :return: self
        """
        hdu = fits.open(filename)[0]  # 该方法使用fits.open函数打开文件，并获取第一个HDU（Header Data Unit）。

        self = cls(filename.split(os.path.sep)[-1],
                   hdu.data.astype(np.float), hdu.header)  # 创建一个AllskyImage实例，将文件名、图像数据和头文件作为参数传递给构造函数，并返回实例。

        try:
            self.datetime = Time(self.header['DATE-OBS'], format='isot')
        except (ValueError, KeyError):
          #  print("No Datatime Info for img: ", filename)
            conf.logger.warning(('No time information for image file '
                                '{}.').format(filename))  # 如果文件中没有日期-观测时间（DATE-OBS）的信息，将会抛出异常并记录警告信息。
            pass
            
        return self

    def write_fits(self, filename):
        """Write `~AllskyImage` instance to FITS image file"""
        hdu = fits.PrimaryHDU(self.data)
        hdu.writeto(filename, overwrite=True)

    # 类的方法创建缩略图的叠加层。它通过使用self.features中的数据来生成叠加层，并根据overlaytype参数的不同选择生成叠加层的方式
    def create_overlay(self, overlaytype='srcdens', regions=None):  # srcdens
        """Create overlay for thumbnail image. Requires self.subregions to be
        initialized. An overlay is an array with the same dimensions as
        self.data` in which certain subregions get assigned certain values as
        defined by `overlaytype`.  为缩略图创建叠加层。需要初始化 self.subregions。叠加层是一个与 self.data 具有相同维度的数组，其中子区域被分配了由 `overlaytype` 定义的某些值

        :param overlaytype: define data source from `self.features` from which
                            overlay should be generated, default: 'srcdens'
        :param regions: list of length=len(self.subregions), highlights
                        subregions with list element value > 0; requires
                        `overlaytype='subregions'`, default: None

        :return: overlay array
        """
        map = np.zeros(self.data.shape)  # np.zeros

        for i, sub in enumerate(self.subregions):
            if overlaytype == 'srcdens':
                map += sub*self.features['srcdens'][i]
            elif overlaytype == 'bkgmedian':
                map += sub*self.features['bkgmedian'][i]
            elif overlaytype == 'bkgmean':
                map += sub*self.features['bkgmean'][i]
            elif overlaytype == 'bkgstd':
                map += sub*self.features['bkgstd'][i]
            elif overlaytype == 'subregions':
                if regions[i]:
                    map += sub
            else:
                raise AllskyImageError('overlaytype "{}" unknown.'.format(
                    overlaytype))

        map[map == 0] = np.nan  # 图片背景
        return map


      # 将 `AllskyImage` 实例化写入缩放的 png 缩略图图像文件
    def write_image(self, filename, overlay=None, mask=None,
                    overlay_alpha=0.2, overlay_color='binary'):
        """Write `~AllskyImage` instance as scaled png thumbnail image file.

        :param filename: filename of image to be written, relative to cwd
        :param overlay: provide overlay or list of overlays, optional
        :param mask: apply image mask before writing image file
        :param overlay_alpha: alpha value to be applied to overlay
        :param overlay_color: colormap to be used with overlay

        :return: None
        """

        conf.logger.info('writing thumbnail "{}"'.format(filename))

        data = self.data

        if mask is not None:
            norm = ImageNormalize(data[mask.data == 1],
                                  conf.THUMBNAIL_SCALE(),
                                  stretch=LinearStretch())
            data[mask.data == 0] = 0
        else:
             norm = ImageNormalize(data, conf.THUMBNAIL_SCALE(),
                              stretch=LinearStretch())

        # create figure
        f, ax = plt.subplots(figsize=(conf.THUMBNAIL_WIDTH,
                                      conf.THUMBNAIL_HEIGHT))


        img = ax.imshow(data, origin='lower',
                        norm=norm, cmap='gray',
                        extent=[0, self.data.shape[1],
                                0, self.data.shape[0]])


        if overlay is not None:
            if not isinstance(overlay, list):
                overlay = [overlay]
                overlay_color = [overlay_color]
            overlay_img = []
            for i in range(len(overlay)):
                overlay_img.append(ax.imshow(overlay[i], cmap=overlay_color[i],
                                             origin='lower', vmin=0,
                                             alpha=overlay_alpha,
                                             extent=[0, overlay[i].shape[1],
                                                     0, overlay[i].shape[0]]))
                overlay_img[i].axes.get_xaxis().set_visible(False)
                overlay_img[i].axes.get_yaxis().set_visible(False)


        plt.axis('off')
        img.axes.get_xaxis().set_visible(False)
        img.axes.get_yaxis().set_visible(False)

        # save thumbnail image
        plt.savefig(filename, bbox_inches='tight',dpi=300 ,pad_inches=0.1)    # pad_inches=0.1  ,alpha=0.8  dpi=conf.THUMBNAIL_DPI
        plt.close()

        # let thumbfilename consist of <night>/<filename>
        self.thumbfilename
        ame = os.path.join(*filename.split(os.path.sep)[-2:])

    def apply_mask(self, mask):
        """Apply `~AllskyImage` mask to this instance"""
        self.data = self.data * mask.data
        
    def crop_image(self):
        """Crop this `~AllskyImage` instance to the ranges defined by
        ``conf.X_CROPRANGE`` and ``conf.Y_CROPRANGE``.
        """
        self.data = self.data [conf.Y_CROPRANGE[0]:conf.Y_CROPRANGE[1],
                              conf.X_CROPRANGE[0]:conf.X_CROPRANGE[1]]


    def extract_features(self, subregions, mask=None):  #subregions
        """Extract image features for each subregion. Image should be cropped
        and masked.

        :param subregions: subregions to be used
        :param mask: mask to be applied in source extraction, optional

        :return: None
        """
        extract_features_start = time.time()
        # set internal pixel buffer 设置内部像素缓冲区
        sep.set_extract_pixstack(20000000)
        # extract time from header and derive frame properties  从主单元中提取时间并导出帧属性
        # t = Time(times, format='isot', scale='utc')
        '''format里面可选iso,isot,jd等，其中iso是普通的年月日，isot是普通的年月日时分秒，jd是儒略日
        这保留原有写法，试着删掉时间信息，看能否跑通'''
        # try:
        #     time = Time(self.header['DATE-OBS'], format='isot')
        #     features = OrderedDict([
        #         ('time', time.isot),
        #         ('filename', self.filename.split(os.path.sep)[-1]),
        #         ('moon_alt', observatory.moon_altaz(time).alt.deg),
        #         ('sun_alt', observatory.sun_altaz(time).alt.deg),
        #         ('moon_phase', 1-observatory.moon_phase(time).value/np.pi),
        #     ])
        # except KeyError as e:
        #     conf.logger.error('missing time data in file {}: {}.'.format(
        #         self.filename, e))
        #     print("Missing time data. ")
        #     return False
        '''在 try 代码块中，定义一个有序字典 features，其中只包含一个键值对，键为 'filename'，值为通过 self.filename 分割后的列表中的最后一个元素。这个操作的目的是获取文件名中的文件名称部分，并将它存储在 features 字典中。
        
        如果 self.filename 中不存在路径分隔符（os.path.sep），那么 split(os.path.sep) 将返回一个只包含一个元素的列表，该元素即为文件名本身。
        
        如果 self.filename 中存在路径分隔符，将其拆分为多个子字符串，并选择最后一个元素作为文件名。'''
        try:
            features = OrderedDict([
                ('filename', self.filename.split(os.path.sep)[-1]),
            ])
        except KeyError as e:
            conf.logger.error('missing time data in file {}: {}.'.format(
                self.filename, e))
        #    print("Missing time data. ")
            return False

        # derive and subtract sky background 导出与减去天空背景
        bkg = sep.Background(self.data.astype(np.float64),
                             bw=conf.SEP_BKGBOXSIZE, bh=conf.SEP_BKGBOXSIZE,
                             fw=conf.SEP_BKGXRANGE, fh=conf.SEP_BKGYRANGE)
        data_sub = self.data - bkg.back()
        print('data sub:',self.data.shape,data_sub.shape,bkg.back().shape)
        # plt.imshow(data_sub, interpolation='nearest', cmap='gray', origin='lower')
        # plt.colorbar()
        # plt.show()

        #  如果提供了掩码，则使用np.ma.array()函数创建了一个掩码数组，并将输入图像数据data_sub中的那些在掩码中为True的像素位置掩盖掉。然后，使用np.ma.median()函数计算了剩余像素的中值，作为基准阈值的一部分。
        # 同时，在计算阈值时，还加上了背景噪声的RMS（均方根）值与一个参数conf.SEP_SIGMA的乘积。
        # 由于SEP库是基于噪声特征来检测目标的，因此这个参数会影响检测灵敏度，越大则检测越宽松，越小则检测越严格。
        # 如果没有提供掩码，则直接使用bkg.globalrms作为阈值基准值，即全局均方根。
        # 最终，根据计算得到的阈值，使用sep.extract()函数从输入图像data_sub中提取出目标。其中，minarea参数用于控制提取目标的最小面积，deblend_nthresh和deblend_cont参数用于进行目标分离（de-blending）操作。
        if mask is not None:
            threshold = (np.ma.median(np.ma.array(data_sub,
                                                  mask=(1-mask))) +
                         np.median(bkg.rms())*conf.SEP_SIGMA)
            src = sep.extract(data_sub, threshold, minarea=conf.SEP_MINAREA,
                              mask=(1-mask),
                              deblend_nthresh=conf.SEP_DEBLENDN,
                              deblend_cont=conf.SEP_DEBLENDV)
        else:
            threshold = (np.median(data_sub) +
                     np.median(bkg.rms())*conf.SEP_SIGMA)
            src = sep.extract(data_sub, 2, minarea=conf.SEP_MINAREA,
                          mask=mask,  # mask
                          deblend_nthresh=conf.SEP_DEBLENDN,
                          deblend_cont=conf.SEP_DEBLENDV)  # threshold

        # apply max_flag cutoff (reject flawed sources)
        src = src[src['flag'] <= conf.SEP_MAXFLAG]
        extract_features_end = time.time()
       # print('281 lines feature extraction time:', extract_features_end-extract_features_start)
        # feature extraction per subregion
        features['srcdens'] = []
        features['bkgmedian'] = []
        features['bkgmean'] = []
        features['bkgstd'] = []
        features['median'] = []  # value of cloud or clear sky
        features['mean'] = []
        features['std'] = []
        # features['variance'] = []
        # features['contrast'] = []

        for i, sub in enumerate(subregions): # subregions
           # print("sub:",sub.shape)

            features['srcdens'].append(len(
                src[sub[src['y'].astype(np.int),
                        src['x'].astype(np.int)]]) / np.sum(sub[mask == 1]))
           # features['srcdens'].append(len(src[sub[src['y'].astype(np.int), src['x'].astype(np.int)]])/np.sum(sub[mask== 1]) )   # (len(src))
            features['bkgmedian'].append(np.median(bkg.back()[sub]))
            features['bkgmean'].append(np.mean(bkg.back()[sub]))
            features['bkgstd'].append(np.std(bkg.back()[sub]))  # (src[0])
            features['median'].append(np.median(self.data[sub]))  # value of cloud or clear sky
            features['mean'].append(np.mean(self.data[sub]))  # data_sub[sub]
            features['std'].append(np.std(self.data[sub]))
            # # 计算图像方差
            # features['variance'].append(np.var(self.data[sub]))
            # # 计算全局对比度
            # features['contrast'].append(((np.max(self.data[sub])-np.mean(self.data[sub]))/np.mean(self.data[sub])))
            np.seterr(divide='ignore', invalid='ignore')  # 消除被除数为0的警告

            '''灰度共生矩阵'''
            # # 灰度值归一化
            # gray_levels = 256
            # image = (sub / np.max(sub)) * (gray_levels - 1)
            # image = image.astype(np.uint8)
            #
            # # 灰度共生矩阵参数
            # d = 1  # step
            # theta = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
            # levels = gray_levels
            #
            # # 计算灰度共生矩阵
            # glcm = np.zeros((levels, levels, len(theta)), dtype=np.uint32)
            # for i in range(image.shape[0] - d):
            #     for j in range(image.shape[1] - d):
            #         for k, angle in enumerate(theta):
            #             x, y = i + d * np.cos(angle), j + d * np.sin(angle)
            #             if x >= 0 and x < image.shape[0] and y >= 0 and y < image.shape[1]:
            #                 i1, j1 = image[i, j], image[int(x), int(y)]
            #                 glcm[i1, j1, k] += 1
            #
            # # 归一化灰度共生矩阵
            # glcm = glcm.astype(np.float64)
            # for k in range(len(theta)):
            #     glcm[:, :, k] /= np.sum(glcm[:, :, k])
            #
            # # 计算灰度共生矩阵统计特征
            # contrast = np.sum(np.square(np.arange(levels) - np.mean(glcm))) * np.sum(
            #     glcm)  # 对比度反应了图像的清晰度和纹理的沟纹深浅。纹理越清晰反差越大对比度也就越大
            # dissimilarity = np.sum(np.abs(np.arange(levels) - np.mean(glcm))) * np.sum(glcm)
            # homogeneity = np.sum(glcm / (1 + np.square(np.arange(levels).reshape(1, -1, 1) - np.mean(glcm))), axis=(0, 1))
            # energy = np.sum(np.square(glcm))  # 角二阶矩（能量），图像灰度分布均匀程度和纹理粗细的度量。当图像纹理均一规则时，能量值较大；反之灰度共生矩阵的元素值相近，能量值较小。
            # entropy = -np.sum(
            #     glcm * np.log2(glcm + 1e-10))  # 熵（Entropy, ENT）度量了图像包含信息量的随机性，表现了图像的复杂程度。当共生矩阵中所有值均相等或者像素值表现出最大的随机性时，熵最大。
            # inverse_difference = np.sum(glcm / (1 + np.abs(np.arange(levels).reshape(1, -1, 1) - np.mean(glcm))),
            #                             axis=(0, 1))  # 逆差距：反映了纹理的清晰程度和规则程度，纹理清晰、规律性较强、易于描述的，值较大。
            #
            # # 输出灰度共生矩阵统计特征
            # # print('Contrast:', contrast)  # 对比度
            # # print('Dissimilarity:', dissimilarity)  # 差异性
            # # print('Homogeneity:', homogeneity)  # 均匀性（向量）
            # # print('Energy:', energy)  # 能量
            # # print('Entropy:', entropy)  # 熵
            # # print('Difference', inverse_difference)  # 向量
            # features['energy'].append(energy)
            # features['entropy'].append(entropy)

        self.subregions =subregions # subregions
        self.features = features

    def write_to_database(self):  # 用于将提取的特征写入数据库
        """Write extracted features to database."""
        session = requests.Session()
        post_headers = {'Content-Type': 'application/json'}  # 创建一个会话对象session，并设置请求头post_headers的内容类型为'application/json'。

        try:  # 通过try-except语句块获取要写入数据库的特征数据data。data是一个字典，包含了要写入的各个特征字段，如日期（'date'）、夜晚标识（'night'）、缩略图文件路径（'filearchivepath'）等。
            data = {'date': self.features['time'],
                    'night': int(self.datetime.strftime('%Y%m%d')),
                    'filearchivepath': self.thumbfilename,
                    'moonalt': self.features['moon_alt'],
                    'sunalt': self.features['sun_alt'],
                    'moonphase': self.features['moon_phase'],
                    'srcdens': self.features['srcdens'],
                    'bkgmean' : self.features['bkgmean'],
                    'bkgmedian' : self.features['bkgmedian'],
                    'bkgstd': self.features['bkgstd'],
            }
        except KeyError:  # 在try块中，将特征数据按照指定的格式填入data字典。其中self.features是一个包含特征数据的字典，通过键名进行访问。

                        # 如果在try块中发生KeyError异常，表示特征数据不完整，记录错误信息并返回None。
            conf.logger.error('data incomplete for file {}; reject.'.format(
                self.filename))
            return None

        post_request = session.post(
            conf.DB_URL+'data/Unlabeled/',
            headers=post_headers, auth=(conf.DB_USER, conf.DB_PWD),
            json=data)  # 使用会话对象session发送POST请求post_request，将特征数据data以JSON格式上传到指定的数据库URL。请求头中包含认证信息，使用conf.DB_USER和conf.DB_PWD作为用户名和密码进行认证。

             # 检查POST请求的响应状态码，如果不是成功（200 OK）或创建成功（201 Created），记录错误信息并抛出ServerError异常。

        if not ((post_request.status_code == requests.codes.ok) or
        (post_request.status_code == requests.codes.created)):
            conf.logger.error('upload to database failed with code {}; {}'.format(
                post_request.status_code, post_request.text))
            raise ServerError('upload to database failed with code {}'.format(
                post_request.status_code))

class AllskyCamera():  # AllskyCamera的类，用于处理全天相机的数据。

    """Class for handling data from an all-sky camera."""

    def __init__(self, night=None):  # 在类的初始化方法__init__()中，接受一个night参数（表示夜晚日期或目录名称）


        if night is None:       # date of the night or directory name
            self.night = self.get_current_night() # 如果night参数为None，则调用get_current_night()方法获取当前夜晚的日期作为默认值；否则使用传入的night值。
        else:  # # 初始化实例变量：self.night表示夜晚日期，self.imgdata表示图像数据数组，默认为None，self.maskdata表示掩码数据数组，默认为None
        # self.subregions表示子区域数组，默认为None，self.polygons表示子区域轮廓多边形，默认为None。
            self.night = night
        self.imgdata = None # image data array  None
        self.maskdata = None    # mask data array
        self.subregions = None  # subregion array
        self.polygons = None    # subregion outline polygons(子区域轮廓多边形)

        conf.update_directories(self.night)  # update_directories()是一个conf对象的方法，用于更新目录结构，根据夜晚日期创建或更新相应的目录。

        conf.logger.info('setting up analysis for night {}'.format(night)) # 在初始化方法中，使用conf.logger记录日志信息，指示正在设置分析程序以处理指定夜晚的数据。
        
    def get_current_night(self, previous_night=False): # get_current_night()是一个类方法，用于根据当前时间获取当前夜晚或随后一夜的目录名称。

        """derive current night's or following night's directory name based on
        current time"""
        if not previous_night: # 如果previous_night参数为False（默认值），则使用当前时间；
            now = datetime.datetime.utcnow()
        else:
            now = (datetime.datetime.utcnow() - datetime.timedelta(days=1)) # 如果为True，则使用当前时间减去一天的时间间隔。返回格式化后的日期字符串（'%Y%m%d'）作为结果。
            
        return now.strftime('%Y%m%d')
        
        
    def download_latest_data(self):
        """Download latest data from camera computer's self.night directory
        using rsync."""

        # build rsync command
        commandline = 'rsync -avr {}:{} {}'.format(
            conf.CAMHOST_NAME,
            os.path.join(conf.CAMHOST_BASEDIR, self.night,
                         '*.{}'.format(conf.FITS_SUFFIX)), conf.DIR_RAW)

        # download data
        conf.logger.info('attempting download with %s', commandline)
        try:
            run = subprocess.Popen(shlex.split(commandline),
                                   close_fds=True)
            run.wait()
        except Exception as e:
            conf.logger.error('download failed: {}'.format(e))
        else:
            conf.logger.info('download succeeded')


    def get_latest_unlabeled_date(self, night):
        """Retrieve latest unlabeled image for a given night from
        database."""
        session = requests.Session()
        get_request = session.get(
            conf.DB_URL+'latest_unlabeled', params={'night': night},
            auth=(conf.DB_USER, conf.DB_PWD))

        conf.logger.info('retrieving latest update of unlabeled database')
        
        if not ((get_request.status_code == requests.codes.ok) or
                (get_request.status_code == requests.codes.created)):
            conf.logger.error("error retrieving latest unlabeled date.",
                              get_request.text, get_request.status_code)
            return None
        
        return get_request.json()
        
    def read_data_from_directory(self, only_new_data=True, crop=True,
                                 batch_size=None, last_image_idx=None): # 从目录中读取图像数据。
        # only_new_data：仅考虑尚未处理过的数据，默认为True。
        # crop：是否裁剪图像，默认为True。
        # batch_size：一次处理的图像数量，默认为None。
        # last_image_idx：从哪个图像索引开始处理，默认为None。
        """Read in images from a directory.

        :param only_new_data: only consider data that have not yet been
                              processed.
        :param crop: crop images
        :param batch_size: how many images to process in one batch
        :param last_image_idx: index of image where to start processing

        """

        conf.logger.info('reading data from directory "{}"'.format(
            conf.DIR_RAW))  # 日志中记录正在从图像目录中读取的图像数据信息

        last_image_night = self.night  # 初始化变量为当前夜晚日期
        if only_new_data:
            last_image = self.get_latest_unlabeled_date(self.night)  # True，获取最新的未标记日期图像
            if last_image is None:   # False，设置索引为0
                last_image_idx = 0
            else:
                last_image_night = last_image['night'] # 否则，将last_image_night更新为last_image的夜晚日期，并检查last_image_night是否存在于目录路径（conf.DIR_RAW）中。

                if (str(last_image_night) in conf.DIR_RAW and
                    last_image_idx is None): # 如果last_image_idx为None，则从last_image的filearchivepath中提取文件索引
                    last_image_idx = int(last_image['filearchivepath'].split(
                        os.path.sep)[-1][len(conf.FITS_PREFIX):
                                         -len(conf.FITS_SUFFIX)-1])  #  去掉前缀conf.FITS_PREFIX和后缀conf.FITS_SUFFIX，并将其转换为整数类型。

            conf.logger.info(('  ignore frames in night {} with indices lower '
                              'than {}.').format(
                                  last_image_night, last_image_idx))  # 日志记录忽略低于指定索引的夜晚图像信息

        data = [] # 存储读取的图像数据的列表
        for i, filename in enumerate(sorted(os.listdir(conf.DIR_RAW))):  # 遍历图片文件名进行排序，便于锁定图像文件
            # check file type
            if (not filename.startswith(conf.FITS_PREFIX) or
                not filename.endswith(conf.FITS_SUFFIX)):
                continue  # 检查图像文件类型，不是指定的前缀名或者后缀名则跳过

            file_idx = str(filename[len(conf.FITS_PREFIX):
                                    -len(conf.FITS_SUFFIX)-1])  #  # 检查图像文件类型，不是fits格式则跳过

            if only_new_data:  # 提取新的图像文件索引，索引小于上一张索引则不是新图像，跳过
                if file_idx <= last_image_idx:
                    continue

            img = AllskyImage.read_fits(os.path.join(conf.DIR_RAW, filename)) # 使用AllskyImage类的read_fits方法读取.fits文件，
            # 并将其路径传递给os.path.join(conf.DIR_RAW, filename)。读取的图像数据存储在变量img中。


            # 这里会因为无时间信息，continue跳过
            # if img.datetime is None:
            #     continue

            # check solar elevation
            # if observatory.sun_altaz(img.datetime).alt.deg >= -6:
            #     continue
            
            if crop: # 图片进行裁剪，日志中记录裁剪信息，并调用img对象的crop_image裁剪图像
                conf.logger.info('  cropping {}'.format(filename))
                img.crop_image()

            data.append(img)  # img对象添加到列表，存储读取的所有图像数据
            
            if batch_size is not None and len(data) >= batch_size:
                break  # 达到或超过指定的图像批量数跳出循环

        conf.logger.info('{} images read for processing'.format(len(data))) # 记录读取并处理的图像数据
        print('565 lines:read_data_from_directory:',data)
        self.imgdata = data
        

    def generate_mask(self, mask_gt=None, mask_lt=None,
                  return_median=False, convolve=None,
                  gaussian_blur=None, filename='mask.fits'):
        """Generate an image mask to cover the horizon.


        :return: mask array (0 is masked, 1 is not masked)
        """
        mask= np.median([img.data for img in self.imgdata], axis=0) #将self.imgdata中所有图像数据的中值计算出来并赋值给变量mask的意义在于生成一个代表图像集合总体特征的掩膜。
                                                                    # 通过计算中值，可以有效地减少图像中的异常值的影响，得到一个更加稳定和可靠的掩膜结果。

        print('588 lines self.imgdata:',self.imgdata)
        print('589 lines mask:',mask)
        conf.logger.info('generating image mask from {} images.'.format(len(
            self.imgdata)))  # 日志记录从多少张图像中生成掩膜

        if gaussian_blur is not None: # 如果指定了高斯模糊参数，使用指定高斯滤波器进行mask模糊处理
            # 有些图像中可能存在非常小的细节或噪声，这些细节可能对于生成掩膜而言并不重要。通过应用高斯滤波器，可以将这些小特征模糊掉，从而在后续处理中减少冗余数据。这有助于提高计算效率和去除不必要的细节。

            conf.logger.info(('  apply gaussian blur with kernel size {'
                              '}.').format(gaussian_blur))
            mask = gaussian_filter(mask, gaussian_blur)

            if not return_median:
                newmask = np.ones(mask.shape)  # False，创建一个与mask形状相同的新掩模数组，初始化全为1
                if mask_gt is not None: # 指定了mask_gt参数的话，低于该参数的都是0.去除图像中较低亮度的像素
                    conf.logger.info(('  mask pixels with values greater '
                                      'than {}.').format(mask_gt))
                    newmask[mask < mask_gt] = 0
                elif mask_lt is not None:  # 指定了参数，高于该参数的都是0。通过阈值过滤来排除掩膜中较亮的部分。这样可以帮助去除图像中较高亮度的像素，使得掩膜更专注于特定的区域或目标
                    conf.logger.info(('  mask pixels with values less '
                                      'than {}.').format(mask_lt))

                    newmask[mask > mask_lt] = 0
                mask = newmask

            if convolve is not None:
                conf.logger.info(('  convolve mask with kernel '
                                  'size {}.').format(convolve))
                mask = np.clip(convolve2d(mask, np.ones((convolve, convolve)),
                                          mode='same'), 0, 1)

        # masked regions have value 1, unmasked regions value 0
        mask2 = mask+1  # mask=mask+1
        print(mask2,mask)
        mask[mask==2] =np.float64(0)

        mask = AllskyImage('mask', mask, {})
        mask.write_fits(os.path.join(conf.DIR_ARCHIVE, filename))

        conf.logger.info('  mask file written to {}.'.format(
            os.path.join(conf.DIR_ARCHIVE, filename)))

        return mask


    def read_mask(self, filename):
        """Read in mask FITS file."""
        if filename is None:
            filename = conf.MASK_FILENAME
        conf.logger.info('read mask image file "{}"'.format(filename))
        self.maskdata = AllskyImage.read_fits(filename)
        
    def process_and_upload_data(self, no_upload=False):
        # 是遍历图像数据列表 self.imgdata，方便对图像进行提取特征、保存缩略图等特定操作，可选择将数据上传到数据库，并返回最后一个文件的索引。
        """Wrapper method to automatically process images and upload data to
        the database. This method also creates thumbnail images that are
        saved to the corresponding archive directory."""
        conf.logger.info('processing image files')  # 记录一条信息，表示正在处理图像文件

        for dat in self.imgdata:  # 遍历 self.imgdata 中的每个元素
            conf.logger.info('extract features from file "{}"'.format(
                dat.filename))  # 记录一条信息，表示从文件中提取特征。dat.filename 是当前元素的文件名。
            file_idx = ''
            # file_idx = str(dat.filename[len(conf.FITS_PREFIX):
            #                             -len(conf.FITS_SUFFIX)-1])  # 从文件名中提取出索引信息，并转换为字符串类型，存储在变量 file_idx 中
            extraction = dat.extract_features(self.subregions,mask=self.maskdata.data # self.maskdata.data None
                                              )  #  subregions 从当前元素中提取特征，并传递 self.subregions 和 self.maskdata.data 参数进行处理。该方法会返回特征的提取结果，存储在变量 extraction 中

            if extraction is False:  # extraction 的值为 False，则使用 conf.

                # logger.error('ignore results for image "{}".'.format(dat.filename)) 记录一条错误信息，表示忽略该图像的处理结果，并继续下一次循环。
                conf.logger.error('ignore results for image "{}".'.format(
                    dat.filename))
                continue

            filename = dat.filename.split(conf.FITS_SUFFIX)[0] + 'png'
            # filename = dat.filename[:dat.filename.find(conf.FITS_SUFFIX)]+'png'   # 将当前元素的图像数据写入到归档目录下，保存为 PNG 格式的缩略图。
            
            dat.write_image(os.path.join(conf.DIR_ARCHIVE, filename),
                            mask=self.maskdata)  # 如果 no_upload 的值不为 True（即默认为 False），则调用 dat.write_to_database() 将当前元素的数据写入数据库。

            if not no_upload:
                dat.write_to_database()

        return file_idx   # 最后，使用 return file_idx 返回变量 file_idx 的值。

    # def generate_subregions(maskdata, target_size=(220, 220)):
    #     """Create subregions array. This array consists of N_subregions
    #     arrays, each with the same dimensions as maskdata.
    #     """
    #     shape = np.array(maskdata.data.shape)
    #     print("shape", shape)
    #     center_coo = shape // 2
    #
    #     n_subregions = conf.N_RINGS * conf.N_RINGSEGMENTS + 1
    #     n_rings = conf.N_RINGS
    #     n_ring_segments = conf.N_RINGSEGMENTS
    #
    #     radius_borders = np.linspace(0, min(shape) // 2, n_rings + 2)
    #     azimuth_borders = np.linspace(-np.pi, np.pi, n_ring_segments + 1)
    #
    #     y, x = np.indices(shape)
    #     r_map = np.sqrt((x - center_coo[0]) ** 2 + (y - center_coo[1]) ** 2).astype(np.int)
    #     az_map = np.arctan2(y - center_coo[1], x - center_coo[0])
    #
    #     subregions = np.zeros([n_subregions, *shape], dtype=np.bool)
    #     polygons = []
    #
    #     subregions[0][(r_map < radius_borders[1])] = True
    #     contours = measure.find_contours(subregions[0], 0.5)
    #     polygons.append((contours[0][:, 0][::10], contours[0][:, 1][::10]))
    #
    #     for i in range(1, n_rings + 1):
    #         for j in range(n_ring_segments):
    #             subregions[(i - 1) * n_ring_segments + j + 1][
    #                 ((r_map > radius_borders[i]) &
    #                  (r_map < radius_borders[i + 1]) &
    #                  (az_map > azimuth_borders[j]) &
    #                  (az_map < azimuth_borders[j + 1]))
    #             ] = True
    #             contours = measure.find_contours(subregions[(i - 1) * n_ring_segments + j + 1], 0.5)
    #             polygons.append((contours[0][:, 0][::10], contours[0][:, 1][::10]))
    #
    #     subregions_resized = np.zeros([n_subregions, *target_size], dtype=np.bool)
    #     for i in range(n_subregions):
    #         subregions_resized[i] = zoom(subregions[i].astype(float),
    #                                      (target_size[0] / shape[0], target_size[1] / shape[1]), order=0).astype(
    #             bool)
    #
    #     return subregions_resized, polygons
    #     subregions_resized, polygons = generate_subregions(maskdata.data.shape)


    def generate_subregions(self): # 创建与掩膜尺寸一致的子区域数组：计算掩膜和边界值生成与掩膜大小一致的子区域组，并得到每个子区域边界轮廓位置
        """Create subregions array. This array consists of N_subregions
        arrays, each with the same dimensions as self.maskdata.
        """
        # '''初始化圆心、半径、角度、子区域数量'''

        shape = np.array(self.maskdata.data.shape)  # np.array(self.maskdata.data.shape)# np.array([300,300])   # 获取掩膜数据尺寸大小，并存在shape中
        print("shape",shape)
        center_coo = shape/2  # 计算掩膜中心坐标
        radius_borders = np.linspace(0, min(shape)/2,
                                     conf.N_RINGS + 2)  # 根据指定的环数生成一系列半径边界值，将掩膜数据划分为多个环状区域
        azimuth_borders = np.linspace(-np.pi, np.pi,
                                      conf.N_RINGSEGMENTS + 1) # 根据指定环状区域分割数，生成方位角边界值
        n_subregions = conf.N_RINGS*conf.N_RINGSEGMENTS+1 # 计算总的子区域数量


        # build templates for radius and azimuth
        y, x = np.indices(shape)  # 创建一个与掩膜尺寸相同的坐标网格。
        r_map = np.sqrt((x-center_coo[0])**2 +
                        (y-center_coo[1])**2).astype(np.int)  # r_map：根据中心坐标center_coo和像素点的坐标(x，y)，通过欧几里得距离公式计算每个像素点相对于中心坐标的半径，构成半径图像
        # 这通过使用NumPy的sqrt函数来计算平方根，并使用astype(np.int)将结果转换为整数类型。

        az_map = np.arctan2(y-center_coo[1],
                            x-center_coo[0])  # az_map计算：根据中心坐标center_coo和像素点的坐标(x，y)，通过通过arctan2函数计算每个像素点相对于中心坐标的方位角，构成方位角图像。
                                                # arctan2函数接受两个参数，分别表示纵坐标的差值和横坐标的差值，返回的结果是一个介于[-π, π]的角度值。
        # r_map和az_map分别表示了图像中每个像素点相对于中心坐标的半径和角度。

        # subregion maps
        # subregions = np.zeros([n_subregions, shape[0],shape[1]], dtype=np.bool)
        subregions = np.zeros([n_subregions, *shape], dtype=np.bool)  # 创建布尔类型数组，存储子区域信息，初始信息都为False

        # polygons around each source region in original image dimensions  原始图像尺寸中每个源区域周围的多边形
        polygons = [] # 存储子区域顶点坐标

        subregions[0][(r_map < radius_borders[1])] = True  # 确定第一个子区域，半径小于第一个子区域半径的为True
        # find contours
        contours = measure.find_contours(  # measure.find_contours检测第一各自区域边缘轮廓，存储在contours变量中
            subregions[0], 0.5)
        # 将每个子区域的顶点列表添加到列表 polygons 中，分别为x y 采样点个数
        polygons.append((contours[0][:, 0][::1],
                         contours[0][:, 1][::1]))
        '''两套嵌套循环分别用于遍历等环数和等经线数量，确定子区域顶点位置'''

        generate_subregions_start = time.time()
        for i in range(1,conf.N_RINGS+1):  # 遍历循环每个环数  先小后大嵌套法则，节省资源
            for j in range(conf.N_RINGSEGMENTS): # 当前环数下，循环遍历每个环状区域分割数
                subregions[(i-1)*conf.N_RINGSEGMENTS+j+1][   #
                    ((r_map > radius_borders[i]) &
                     (r_map < radius_borders[i+1]) &
                     (az_map > azimuth_borders[j]) &
                     (az_map < azimuth_borders[j+1]))] = True  # 将位于指定半径和方位角的像素标为True，表示属于当前子区域
                contours = measure.find_contours(
                    subregions[(i-1)*conf.N_RINGSEGMENTS+j+1], 0.5)  # 寻找当前子区域边界轮廓，存储在contour变量中

                polygons.append((contours[0][:,0][::10],
                                 contours[0][:,1][::10]))  # 将轮廓线每隔10个像素的顶点坐标加入到顶点列表
        generate_subregions_end = time.time()
        print(' generate_subregions time:', generate_subregions_end - generate_subregions_start)
                # polygons1=np.array(polygons)
                # print('polygons:',polygons1.shape)
        '''-------------------------------------------原始---------------------------------'''
        # '''两套嵌套循环分别用于遍历等环数和等经线数量，这两个数值越大，循环次数越多，耗时增加'''
        # for i in range(conf.N_RINGS+1):
        #     for j in range(conf.N_RINGSEGMENTS):
        #         if i == 0 and j==0:
        #             subregions[0][(r_map < radius_borders[i+1])] = True
        #             # find contours
        #             contours = measure.find_contours(          # measure.find_contours检测二值图像的边缘轮廓
        #                 subregions[0], 0.5)
        #         elif i==0 and j>0:
        #             break
        #         else:
        #             subregions[(i-1)*conf.N_RINGSEGMENTS+j+1][
        #                 ((r_map > radius_borders[i]) &
        #                  (r_map < radius_borders[i+1]) &
        #                  (az_map > azimuth_borders[j]) &
        #                  (az_map < azimuth_borders[j+1]))] = True
        #             contours = measure.find_contours(
        #                 subregions[(i-1)*conf.N_RINGSEGMENTS+j+1], 0.5)
        #
        #         polygons.append((contours[0][:,0][::10],
        #                          contours[0][:,1][::10]))


        self.subregions = subregions
        print('size:',subregions.shape)
        # self.polygons = np.array(polygons)
        return len(self.subregions)

    
# class LightGBMModel():
#     """Class for use of lightGBM model."""
#
#     def __init__(self):
#         self.data_X = None      # pandas DataFrame
#         self.data_y = None      # pandas DataFrame
#         self.model = None       # model implementation
#         self.filename = None    # model pickle filename
#         self.train_score = None # model training score
#         self.test_score = None  # model test score
#         self.val_score = None   # model validation sample score
#         self.f1_score_val = None  # model validation sample f1 score
#
#     def retrieve_training_data(self, size_limit=None):
#         """Retrieves feature data from webapp database."""
#         n_subregions = conf.N_RINGS*conf.N_RINGSEGMENTS+1
#
#         get = requests.get(conf.TRAINDATA_URL)
#         if get.status_code != requests.codes.ok:
#             raise ServerError('could not retrieve training data from server')
#         raw = pd.DataFrame(get.json())
#
#         data = pd.DataFrame()
#         for j in range(len(raw['moonalt'])):
#             frame = pd.DataFrame(OrderedDict(
#                 (('moonalt', [raw['moonalt'][j]]*n_subregions),
#                  ('sunalt', [raw['sunalt'][j]]*n_subregions),
#                  ('moonphase', [raw['moonphase'][j]]*n_subregions),
#                  ('subid', range(n_subregions)),
#                  ('srcdens', raw['srcdens'][j]),
#                  ('bkgmean', raw['bkgmean'][j]),
#                  ('bkgmedian', raw['bkgmedian'][j]),
#                  ('bkgstd', raw['bkgstd'][j]),
#                  ('srcdens_3min', raw['srcdens_3min'][j]),
#                  ('bkgmean_3min', raw['bkgmean_3min'][j]),
#                  ('bkgmedian_3min', raw['bkgmedian_3min'][j]),
#                  ('bkgstd_3min', raw['bkgstd_3min'][j]),
#                  ('srcdens_15min', raw['srcdens_15min'][j]),
#                  ('bkgmean_15min', raw['bkgmean_15min'][j]),
#                  ('bkgmedian_15min', raw['bkgmedian_15min'][j]),
#                  ('bkgstd_15min', raw['bkgstd_15min'][j]),
#                  ('cloudy', raw['cloudy'][j]))))
#             data = pd.concat([data, frame])
#
#         self.data_X = data.drop(['cloudy'], axis=1)
#         self.data_y = np.ravel(data.loc[:, ['cloudy']].values).astype(int)
#         self.data_X_featurenames = data.drop(['cloudy'], axis=1).columns.values
#
#         # limit data set size to size_limit subregions
#         if size_limit is not None:
#             self.data_X = self.data_X[:size_limit]
#             self.data_y = self.data_y[:size_limit]
#
#         return len(self.data_y)
#
#     def load_data(self, filename):
#         """Load feature data from file."""
#
#         data = pd.read_csv(filename, index_col=0)
#
#         # split features and target
#         self.data_X = data.drop(['cloudy'], axis=1)
#         self.data_y = np.ravel(data.loc[:, ['cloudy']].values).astype(int)  # ravel函数的功能是将原数组拉伸成为一维数组
#         self.data_X_featurenames = data.drop(['cloudy'], axis=1).columns.values
#
#         return len(self.data_y)
#
#     def train(self, parameters=conf.LGBMODEL_PARAMETERS, cv=5):
#         """Train """
#
#         # split data into training and validation sample
#         X_cv, X_val, y_cv, y_val = train_test_split(
#             self.data_X, self.data_y, test_size=0.1, random_state=42)
#
#         # define model
#         lgb = LGBMClassifier(objective='binary', random_state=42,
#                              n_jobs=-1, **parameters)
#         # train model
#         lgb.fit(X_cv, y_cv)
#         self.model = lgb
#
#         # derive cv scores
#         cv_results = cross_validate(lgb, X_cv, y_cv, cv=cv,
#                                     return_train_score=True)
#         self.train_score = np.max(cv_results['train_score'])
#         self.test_score = np.max(cv_results['test_score'])
#         self.parameters = parameters
#         self.val_score = self.model.score(X_val, y_val)
#         self.f1_score_val = f1_score(y_val, self.model.predict(X_val))
#
#         return self.val_score
#
#
#     def train_randomizedsearchcv(self, n_iter=100,
#         distributions=conf.LGBMODEL_PARAMETER_DISTRIBUTIONS,
#         cv=3, scoring="accuracy"):
#         """Train the lightGBM model using a combined randomized
#         cross-validation search."""
#
#         # split data into training and validation sample
#         X_grid, X_val, y_grid, y_val = train_test_split(
#             self.data_X, self.data_y, test_size=0.1, random_state=42)
#
#         # initialize model
#         lgb = LGBMClassifier(objective='binary', random_state=42, n_jobs=-1)
#
#         # initialize random search + cross-validation
#         lgbrand = RandomizedSearchCV(lgb, distributions, cv=cv, scoring=scoring,
#                                      n_iter=n_iter, return_train_score=True)
#
#         # fit model
#         lgbrand.fit(X_grid, y_grid)
#
#         self.cv_results = lgbrand.cv_results_
#         self.model = lgbrand.best_estimator_
#
#         # derive scores
#         self.train_score = lgbrand.cv_results_['mean_train_score'][lgbrand.best_index_]
#         self.test_score = lgbrand.cv_results_['mean_test_score'][lgbrand.best_index_]
#         self.parameters = lgbrand.cv_results_['params'][lgbrand.best_index_]
#         self.val_score = self.model.score(X_val, y_val)
#         self.f1_score_val = f1_score(y_val, self.model.predict(X_val))
#
#         return self.val_score
#
#     def write_model(self,
#                     filename=os.path.join(conf.DIR_ARCHIVE+'model.pickle')):
#         """Write trained model to file."""
#         self.filename = filename
#         dump(self.model, filename)
#
#     def read_model(self,
#                    filename=os.path.join(conf.DIR_ARCHIVE+'model.pickle')):
#         """Read trained model from file."""
#         self.filename = filename
#         self.model = load(filename)
#
#     def predict(self, X):
#         """Predict cloud coverage for feature data."""
#         return self.model.predict(X)

