'''sun.alt,sun.az,moon.alt,moon.az'''

import os
import ephem
import torch.cuda
from ephem import *
from astroplan import Observer
import pandas as pd
from datetime import datetime, timedelta
import math
import exifread
import csv

path_a = "Raw images/"
output_filename = '../scripts/illumination.csv'

# 新建 csv 文件并写入表头
with open(output_filename, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['file_name',  'sun_alt', 'sun_az', 'moon_alt', 'moon_az'])  # 'time',

    for file in os.listdir(path_a):
        # exifread处理图片
        f = open(path_a + "/" + file, "rb")
        tags = exifread.process_file(f)

        # 读取照片拍摄时间
        photo_time = str(tags['EXIF DateTimeOriginal'])

        # 计算时间信息
        t_bj = datetime.strptime(photo_time, '%Y:%m:%d %H:%M:%S')
        t_utc = t_bj - timedelta(hours=8)  # 将北京时间转换为世界时间

        # 计算太阳高度角和方位角
        Xinjiang = ephem.Observer()
        Xinjiang.lat = '38.33044'  # 纬度
        Xinjiang.lon = '74.89676'  # 经度
        Xinjiang.elevation = 4526  # 海拔
        Xinjiang.date = t_utc
        sun = ephem.Sun()
        sun.compute(Xinjiang)
        sun_alt = math.degrees(sun.alt)  # 太阳高度角
        sun_az = math.degrees(sun.az)    # 太阳方位角

        # 计算月亮高度角和方位角
        moon = ephem.Moon()
        moon.compute(Xinjiang)
        moon_alt = math.degrees(moon.alt)  # 月亮高度角
        moon_az = math.degrees(moon.az)    # 月亮方位角

        # 将结果复制37次写入 csv 文件
        for _ in range(37):
            row = [file, sun_alt, sun_az, moon_alt, moon_az]  #  photo_time,
            writer.writerow(row)
# part2_end=time.time()
# print('part 2 time:',part2_end-part2_start)
# print("part2 is finished!")
