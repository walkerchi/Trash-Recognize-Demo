#these config could be customize
DOWNLOAD = True
UNZIP = True   # if you only want to unzip, make sure there is a taco_10_3.zip file in the server fold

# you'd better not change the config below
WEIGHT_PATH = 'server'
WEIGHT_URL = 'https://github.com/pedropro/TACO/releases/download/1.0/taco_10_3.zip'
WEIGHT_NAME = 'mask_rcnn_taco_0100.h5'

import requests
import os
import shutil
import zipfile
import time
from contextlib import closing
from tqdm import tqdm
ZIPFILE_NAME = WEIGHT_URL.split('/')[-1]  #taco_10_3.zip
DST_PATH = os.path.join(WEIGHT_PATH,WEIGHT_NAME)
WEIGHT_NAME = ZIPFILE_NAME.split('.')[0] + '/' + WEIGHT_NAME
SRC_PATH = os.path.join(WEIGHT_PATH, WEIGHT_NAME)
ZIPFILE_PATH = os.path.join(WEIGHT_PATH, ZIPFILE_NAME)


headers = {}
headers['User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/49.0.2623.221 Safari/537.36 SE 2.X MetaSr 1.0'


def down_from_url(url, dst):
    response = requests.get(url, headers=headers, stream=True)  # (1)
    file_size = int(response.headers['content-length']) # (2)
    if os.path.exists(dst):
        first_byte = os.path.getsize(dst)  # (3)
    else:
        first_byte = 0
    if first_byte >= file_size: # (4)
        return file_size

    header = {"Range": f"bytes={first_byte}-{file_size}"}

    pbar = tqdm(total=file_size, initial=first_byte, unit='B', unit_scale=True, desc=dst)
    req = requests.get(url, headers=header, stream=True)  # (5)
    with open(dst, 'ab') as f:
        for chunk in req.iter_content(chunk_size=1024):     # (6)
            if chunk:
                f.write(chunk)
                pbar.update(1024)
    pbar.close()
    return file_size

def download(url, path, filename):
    print('地址：' + url)
    print('开始下载,' + filename)
    start_time = time.time()
    down_from_url(url, filename)
    end_time = time.time()
    print(f"下载完成,共花费了{end_time - start_time}s")


########################################
########################################

# download
if DOWNLOAD:
    download(WEIGHT_URL,WEIGHT_PATH,ZIPFILE_PATH)

#unzip 
if UNZIP:
    f = zipfile.ZipFile(ZIPFILE_PATH, 'r')
    f.extract(WEIGHT_NAME, WEIGHT_PATH)
    f.close()
    shutil.move(SRC_PATH,DST_PATH)    # from server/taco_10_3/xxx.h   to server/xxx.h
    os.remove(os.path.join(WEIGHT_PATH,ZIPFILE_NAME))   # remove server/taco_10_3.zip
    shutil.rmtree(os.path.join(WEIGHT_PATH,ZIPFILE_NAME.split('.')[0]))  # remove server/taco_10_3 folder



