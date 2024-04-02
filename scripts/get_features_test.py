import csv
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import cloudynight
import time
from cloudynight import AllskyCamera

start=time.time()
def main():

    read_satrt=time.time()
    cam = cloudynight.AllskyCamera('F:/github/scripts/imsges2/')
    cam.read_data_from_directory(only_new_data=False)  # read fits
    read_end=time.time()
    print('read a fits image time:',read_end-read_satrt)
    # print(cam.imgdata)
    '''---------------------------------------------------------median and mask----------------------------------------------------------'''
    median = cam.generate_mask(return_median=True, filename='median.fits')
    mask = cam.generate_mask(mask_lt=3400, gaussian_blur=10, convolve=20, filename='mask.fits')
    plt.imshow(mask.data)
    plt.show()
    '''
    # meadian.fits 、mask.fits
    a=median.data
    plt.imshow(a)
    plt.show()
    '''
    mask_satat=time.time()
    cam.read_mask(filename='./workbench/images/mask.fits')
    cam.generate_subregions()
    # print(len(cam.subregions), 'subregions were created.')
    mask_end=time.time()
    print('mask and subregion time:',mask_end-mask_satat)

    '''---------------------------------subregions vision and saving-----------------------------------'''
    for subi in range(len(cam.subregions)):
        print('plotting subregion', subi)
        f, ax = plt.subplots(figsize=(6,6))
        ax.imshow(cam.subregions[subi], origin='lower', vmin=0, vmax=1,cmap='gray')
        # plt.imshow(cam.subregions[subi], origin='lower', vmin=0, vmax=1)
        plt.axis('off')
        plt.savefig(os.path.join(cloudynight.conf.DIR_ARCHIVE,
                                'subregion_{:02d}.png'.format(subi)))  # subregion_{:02d}.png

        plt.close()

    start = time.time()
    cam.process_and_upload_data(no_upload=True)
    end = time.time()
    print(' feature time:', end - start)

    save_dir = "F:/github/scripts"
    csv_path = os.path.join(save_dir, "features.csv")

    print("start saving features now...")

    save_start=time.time()
    with open(csv_path, 'w', encoding='utf8', newline='') as fw:
        cw = csv.writer(fw)
        cw.writerow(["file_name","subregions_id", "bkg_median", "bkg_mean", "bkg_std", "density","object_median","object_mean",
                     "object_std"])
        for img in tqdm(cam.imgdata):
            sourcedens_overlay = img.create_overlay(overlaytype='bkgmedian')
            img.write_image(overlay=sourcedens_overlay, mask=cam.maskdata,
                            filename=os.path.join(cloudynight.conf.DIR_ARCHIVE,
                                        '{}_bkgmedian.png'.format(
                                    img.filename[:img.filename.find('.fit')])))
            plt.imshow(sourcedens_overlay)
            plt.close()
            for idx,(median,mean,std,density,omedian,omean,ostd) in enumerate(zip(img.features["bkgmedian"],img.features["bkgmean"],img.features["bkgstd"],
                                                                       img.features["srcdens"],img.features["median"],img.features["mean"],
                                                                    img.features["std"])):
                # line65：,energy,entropy   line66：,img.features["energy"],img.features["entropy"]
                cw.writerow([img.filename,str(idx),str(median),str(mean),str(std),str(density),
                             str(omedian),str(omean),str(ostd)])  # ,str(energy),str(entropy)
    save_end=time.time()
    print('save features time:',save_end-save_start)

if __name__ == '__main__':
    main()
end=time.time()
print('total time is:',end-start)