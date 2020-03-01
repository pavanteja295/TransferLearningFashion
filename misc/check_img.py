"""
Author : Pavan Teja Varigonda
"""

import os
import argparse
import pandas as pd
import time
import glob
import json
import cv2
from PIL import Image, ImageOps
# from bidict import bidict

# def main():
#     parser = argparse.ArgumentParser(description='Fashion Dataset')
#     parser.add_argument('--dataset', dest='dataset', default='fashion-dataset', type=str,
#                         help=" select among fashion-dataset / fashion-dataset-small")

#     args = parser.parse_args()


#     # select the data path
#     dataset = args.dataset
#     cwd = os.getcwd()
#     datapath = os.path.join(cwd, 'data', dataset)
#     annot_csv = os.path.join(datapath, 'styles.csv')

#     # now for some stats
#     tic = time.time()
#     # some lines are skipped due to incosis columns 
#     csv_ = pd.read_csv(annot_csv, error_bad_lines=False)
#     toc = time.time()
#     print('Time taken to read csv {:.2f} secs'.format(toc - tic))
#     total_val_rows = len(csv_)

#     # check if each product has an image pair 
#     #get all img_ids 
#     img_dir = os.path.join(datapath, 'images')
#     imgs_name = glob.glob(os.path.join(img_dir,  '*.jpg')) +  glob.glob(os.path.join(img_dir,  '*.png'))
#     imgs_name = [int(os.path.basename(img)[:-4]) for img in imgs_name]

#     # total number of rows with images
#     val_rows_ = csv_[csv_['id'].isin(imgs_name)]

#     # remove those with non finite values in  year and class 
#     val_rows_ = val_rows_.dropna(subset=['year', 'articleType'])

#     data_div =  {}

#     # different classes
#     classes_ = val_rows_.groupby(val_rows_['articleType'])
#     class_cnt_ = classes_.size().sort_values(ascending=False)
#     classes_lst = {cls_ : i for i, cls_ in enumerate(list(class_cnt_.index.values))}
#     print(class_cnt_)
#     # import pdb; pdb.set_trace()

#     data_div['class_list'] = classes_lst

#     for dtype_ in ['finetune', 'pretrain']:
        
#         data_div[dtype_] = {}

#         if dtype_ == 'pretrain':
#             # count the 3_4 pretrain
#             class_div = class_cnt_[:20]
#         else:
#             class_div = class_cnt_[20:]
        
        
#         data_div[dtype_]['total_class_count'] = class_div.to_dict()
#         data_div[dtype_]['total_samples'] = int(class_div.sum())
        

#         # data 
#         data_filter_ = val_rows_[val_rows_['articleType'].isin(class_div.index)]

#         # strip down train and test
#         splits = ['test', 'train']
#         for split in splits:
#             mod = 1
#             if split == 'train':
#                 mod_ = 0
#             else:
#                 mod_ = 1

#             data_filter_split = data_filter_[data_filter_['year'].astype('int') % 2 == mod_]

#             data_div[dtype_][split] = {}
#             data_div[dtype_][split]['class'] = [ classes_lst[cls_]  for  cls_ in list(data_filter_split['articleType'].values)]
#             data_div[dtype_][split]['image_list'] = [ os.path.join(img_dir, str(img_) + '.jpg') for img_ in list(data_filter_split['id'].values) ]

#             print('Total {} samples for {} are {}'.format(split, dtype_, len(data_div[dtype_][split]['image_list'])))            
#             assert  len(data_div[dtype_][split]['class'])  == len(data_div[dtype_][split]['image_list'])


#     with open(os.path.join(cwd,'Task1' , 'data' , dataset ,'data.json'), 'w') as fp:
#         json.dump(data_div, fp)


# if __name__ == '__main__':
#     main()

dataset = 'fashion-dataset'
cwd = os.getcwd()
datapath = '/mnt/datasets/datasets/fashion-dataset/fashion-dataset/' #os.path.join(cwd, 'data', dataset) #'/mnt/datasets/datasets/fashion-dataset/fashion-dataset/' #
img_dir = os.path.join(datapath, 'images')
img_dir_rs = os.path.join(datapath, 'images_resized')
imgs_name = glob.glob(os.path.join(img_dir,  '*.jpg'))
img_size = 300
im_lst = []
h_w = set()
ar = set()
print(h_w)
print(ar)
# make all images of size 448 x 448
dict_ = {}
for i, img in enumerate(imgs_name):
    if i % 1000 == 0 :
        print(i)
    # img_np = cv2.imread(img)
    img_np = Image.open(img).convert('RGB')
    # print(img_np.size)
    
    if img_np.size[0] > img_np.size[1]:
        # import pdb; pdb.set_trace()
        min_ = img_np.size[1]
        max_ = int(((300.0 / min_) * img_np.size[0] ))
        img_ = img_np.resize((max_, 300))

    else:
        
        min_ = img_np.size[0]
        max_ = int((300.0 / min_) * img_np.size[1] )
        img_ = img_np.resize((300, max_))


    #import pdb; pdb.set_trace()
    img_name = img[img.rfind('/') + 1:]
    img_n_path = os.path.join(img_dir_rs, img_name)
    # print(img_n_path)
    print(img_np.size, '  changed to', img_.size)
    img_.save(img_n_path)
    # h_w.add(img_np.size)
    # ar.add(img_np.size[0] / img_np.size[1])
    # if img_np.size not in dict_.keys():
    #     dict_[img_np.size] = 1
    # else:
    #     dict_[img_np.size] += 1

# print(h_w)
# print(ar)
# print(dict_)

    
    
    # if not img_np.shape == (80, 60, 3):
    #     print(img_np.shape)
    #     im_lst.append(img)
    
    #     cv2.imwrite(img_rs, img_np)

# im_lst = ['/home/pavanteja/workspace/IntuitionMachines/transfer_learning/data/fashion-dataset-small/images/29519.jpg', '/home/pavanteja/workspace/IntuitionMachines/transfer_learning/data/fashion-dataset-small/images/57730.jpg', '/home/pavanteja/workspace/IntuitionMachines/transfer_learning/data/fashion-dataset-small/images/25958.jpg', '/home/pavanteja/workspace/IntuitionMachines/transfer_learning/data/fashion-dataset-small/images/28092.jpg', '/home/pavanteja/workspace/IntuitionMachines/transfer_learning/data/fashion-dataset-small/images/1799.jpg', '/home/pavanteja/workspace/IntuitionMachines/transfer_learning/data/fashion-dataset-small/images/25299.jpg', '/home/pavanteja/workspace/IntuitionMachines/transfer_learning/data/fashion-dataset-small/images/44101.jpg', '/home/pavanteja/workspace/IntuitionMachines/transfer_learning/data/fashion-dataset-small/images/52166.jpg', '/home/pavanteja/workspace/IntuitionMachines/transfer_learning/data/fashion-dataset-small/images/2311.jpg', '/home/pavanteja/workspace/IntuitionMachines/transfer_learning/data/fashion-dataset-small/images/35915.jpg', '/home/pavanteja/workspace/IntuitionMachines/transfer_learning/data/fashion-dataset-small/images/1800.jpg', '/home/pavanteja/workspace/IntuitionMachines/transfer_learning/data/fashion-dataset-small/images/56624.jpg', '/home/pavanteja/workspace/IntuitionMachines/transfer_learning/data/fashion-dataset-small/images/59593.jpg', '/home/pavanteja/workspace/IntuitionMachines/transfer_learning/data/fashion-dataset-small/images/11151.jpg', '/home/pavanteja/workspace/IntuitionMachines/transfer_learning/data/fashion-dataset-small/images/28492.jpg', '/home/pavanteja/workspace/IntuitionMachines/transfer_learning/data/fashion-dataset-small/images/25943.jpg', '/home/pavanteja/workspace/IntuitionMachines/transfer_learning/data/fashion-dataset-small/images/50908.jpg', '/home/pavanteja/workspace/IntuitionMachines/transfer_learning/data/fashion-dataset-small/images/59606.jpg', '/home/pavanteja/workspace/IntuitionMachines/transfer_learning/data/fashion-dataset-small/images/56128.jpg', '/home/pavanteja/workspace/IntuitionMachines/transfer_learning/data/fashion-dataset-small/images/5408.jpg', '/home/pavanteja/workspace/IntuitionMachines/transfer_learning/data/fashion-dataset-small/images/14776.jpg', '/home/pavanteja/workspace/IntuitionMachines/transfer_learning/data/fashion-dataset-small/images/1801.jpg', '/home/pavanteja/workspace/IntuitionMachines/transfer_learning/data/fashion-dataset-small/images/11160.jpg']
# for im in im_lst:
#     im_np = cv2.imread(im)
#     import pdb; pdb.set_trace()
#     img_nw = cv2.resize(im_np, (80, 60))
#     cv2.imwrite(img_nw, im)