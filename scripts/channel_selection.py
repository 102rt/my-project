'''split the raw images into r、g1、g2、b and covert them to fits'''
import rawpy
import numpy as np
import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import cv2
import os


def reader(path):
    with rawpy.imread('../Raw images/' + path) as raw:  # read from rawexam
        bdr = raw.black_level_per_channel
        # print(raw.raw_image.min())
        # print('black level per channel:', bdr)

        rbdr, g1bdr, bbdr, g2bdr = bdr  # black level per channel
        # print(raw.color_desc,'\n',raw.raw_pattern)     #basic message
        im = np.array(raw.raw_image, dtype=np.int16)  # 原：dtype=np.int32，图像以32位深度存储;raw.raw_image访问原始数据

        # print(im.shape)
        red = im[0::2, 0::2]  # this is rggb,but with a 90 degree transition
        green1 = im[0::2, 1::2]  # use the raw.color_desc to check channel info
        green2 = im[1::2, 0::2]
        blue = im[1::2, 1::2]

        # print("bdr:",bdr)
        red_hist = red - rbdr
        green1_hist = green1 - g1bdr
        green2_hist = green2 - g2bdr
        blue_hist = blue - bbdr


        # # red channel crop
        # red_out = red_hist[193:1603, 630:2060]  # 193:1603,630:2060
        # red_out1 = np.zeros(red_out.shape[:2], dtype=np.uint8)  # height and width of images
        #
        # mask = cv2.circle(red_out1, (716, 706), 570, (255, 0, 255), -11)
        # image_red = cv2.add(red_out, np.zeros(np.shape(red_out), dtype=np.int16), mask=mask)  # dtype=np.int16  , dtype=cv2.CV_16U

        # g1 channel crop
        green1_out = green1_hist[193:1603, 630:2060]  # 193:1603,630:2060
        green1_out1 = np.zeros(green1_out.shape[:2], dtype=np.uint8)
        mask1 = cv2.circle(green1_out1, (716, 706), 570, (255, 0, 255), -11)

        # cv2.destroyAllWindows()
        # cv2.namedWindow("masked image", 0)
        # cv2.resizeWindow("masked image", 700,700)  # 设置窗口大小
        # cv2.imshow("masked image",mask1)
        # cv2.waitKey(0)

        # green1_out = green2_hist[750:1200, 750:1200]
        # green1_out1 = np.zeros(green1_out.shape[:2], dtype=np.uint8)
        # mask1 = cv2.circle(green1_out1, (716, 706), 225, (255, 0, 255), -11)
        image_gr1 = cv2.add(green1_out, np.zeros(np.shape(green1_out), dtype=np.int16), mask=mask1)  # cv2.multiply()
       # image_gr1=image_gr1.astype(np.uint16)
       # print("1",np.min(image_gr1))


        # g2 channel crop
        green2_out = green2_hist[193:1603, 630:2060]
        green2_out2 = np.zeros(green2_out.shape[:2], dtype=np.uint8)
        mask2 = cv2.circle(green2_out2, (716, 706), 570, (255, 0, 255), -11)
        # green2_out = green2_hist[750:1200, 750:1200]
        # green2_out2 = np.zeros(green2_out.shape[:2], dtype=np.uint8)
        # mask2 = cv2.circle(green2_out2, (716, 706), 225, (255, 0, 255), -11)
        image_gr2 = cv2.add(green2_out, np.zeros(np.shape(green2_out), dtype=np.int16), mask=mask2)
        # image_gr2 = image_gr2.astype(np.uint16)
        # print("2",np.min(image_gr2))
        # # histogram
        # plt.hist(image_gr2.flatten(), rwidth=1.8, facecolor='blue')
        # plt.show()

        # # blue channel crop
        # blue_out = blue_hist[193:1603, 630:2060]
        # blue_out1 = np.zeros(blue_out.shape[:2], dtype=np.uint8)
        # mask3 = cv2.circle(blue_out1, (716, 706), 570, (255, 0, 255), -11)
        # image_blue = cv2.add(blue_out, np.zeros(np.shape(blue_out),dtype=np.int16), mask=mask3 )  #,dtype=np.uint16), mask=mask3, dtype=cv2.CV_16U


        # merge
        '''
        r = image_red
        g1 = image_gr1
        g2 = image_gr2
        b = image_blue
        result = cv2.merge((b, g1, g2, r))  # b,g1,g2,r
        # result = cv2.cvtColor(result.astype(np.float32), cv2.COLOR_BGR2GRAY)  # 彩色图像只支持32位浮点数
        result = np.array(result, dtype=np.int16)  # float32转int16 fits格式
        print("fits merge:", result.max(), result.min())
       '''

        c = (image_gr1 + image_gr2) / 2  # color channel selection
        print("c:",c.shape)

    hdu1 = pyfits.PrimaryHDU(c)
    hdu1.writeto('../images2/' + path + 'img.fits', overwrite='True')  # path+'Red.fits'

    # hdu2 = pyfits.PrimaryHDU(image_gr1)  # green1_out
    # hdu2.writeto('E:/Afirstpaper/cloud/CODE/KLCAM/channelfits/'+path+'img.fits',overwrite='True')
    # print("finish!")
    # hdu3 = pyfits.PrimaryHDU(image_gr2)
    # hdu3.writeto('E:/Afirstpaper/cloud/CODE/KLCAM/channelfits/'+path+'Green2.fits',overwrite='True')
    # hdu4 = pyfits.PrimaryHDU(image_blue)
    # hdu4.writeto('E:/Afirstpaper/cloud/CODE/KLCAM/channelfits/'+path+'Blue.fits',overwrite='True')
    # hdu5 = pyfits.PrimaryHDU(result)
    # hdu5.writeto('E:/Afirstpaper/cloud/CODE/KLCAM/channelfits/'+path+'im.fits',overwrite='True')

    return ()


lists = os.listdir('../Raw images/')
for i in range(len(lists)):
    path = lists[i]
    reader(path)