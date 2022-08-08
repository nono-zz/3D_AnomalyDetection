import cv2
import tifffile as tiff
import os
import glob


def depthNormFN(img, depthMax =  0.727695):
    imgNorm = img/depthMax * 255
    return imgNorm
    



dataset_root = '/home/zhaoxiang/3D-ADS/datasets/mvtec3d'
objects = os.listdir(dataset_root)
objects.sort()
trainOrtest = ['test', 'train', 'validation']
fileCategory = 'xyz'

depthMax = 0.727695
depthMin = 0

for object in objects:
    if '.' in object:
        continue
    object_path = os.path.join(dataset_root, object)
    
    # categories = os.listdir(object_path)
    # categories.remove('calibration')
    for folder in trainOrtest:
        subFolder = os.path.join(object_path, folder)
        for defectType in os.listdir(subFolder):
            typeFolder = os.path.join(subFolder, defectType)
            
            imageFolder = os.path.join(typeFolder, fileCategory)
            
            saveFolder = imageFolder.replace('xyz', 'depth')
        # if os.path.exists(saveFolder):
        #     os.rmdir(saveFolder)
            if not os.path.exists(saveFolder):
                os.mkdir(saveFolder)

            for file in glob.glob(os.path.join(imageFolder, '*.tiff')):
                tiff_img = tiff.imread(file)
                depthImg = tiff_img[:,:,2]
                
                
                # depth image normalization
                depthImgNorm = depthNormFN(depthImg)
                
                # save the img
                new_path = file.replace('xyz', 'depth')
                new_path = new_path.replace('tiff', 'png')
                cv2.imwrite(new_path, depthImgNorm)
                
                # if depthMin > depthImg.min():
                #     depthMin = depthImg.min()
                # # depthImg = 

# print(depthMin)
            
            
        
        


# tiff_img = tiff.imread('/home/zhaoxiang/3D-ADS-main/datasets/mvtec3d/dowel/train/good/xyz/002.tiff')
# rgb_img = cv2.imread('/home/zhaoxiang/3D-ADS-main/datasets/mvtec3d/dowel/train/good/rgb/002.png')



# for i in range(tiff_img.shape[2]):
#     cv2.imwrite('tiff_{}.png'.format(i), tiff_img[:,:,i])
# print('done')