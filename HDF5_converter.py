from utils.config import config
from utils.loader import DatasetLoader
from utils.config import config
from utils.preprocess import Preprocessor
from sklearn.model_selection import train_test_split
from utils.io import HDF5DatasetWriter

import numpy as np
import progressbar

paths =  DatasetLoader.load(config.data_path)   

(trainPaths, testPaths)  = train_test_split(paths, test_size=config.NUM_TEST,random_state=42)               
(trainPaths, valPaths) = train_test_split(trainPaths,test_size=config.NUM_VAL,random_state=42) 
   
datasets = [
("train", trainPaths, config.TRAIN_HDF5),
("val", valPaths, config.VAL_HDF5),
("test", testPaths, config.TEST_HDF5)]

for (dType, paths, outputPath) in datasets:

    print("[INFO] building {}...".format(outputPath))
    dim_0 = config.data_shape[0]
    dim_1 = config.data_shape[1]
    dim_2 = config.data_shape[2]
    dim_3 = config.data_shape[3]
    writer = HDF5DatasetWriter((len(paths), dim_0, dim_1, dim_2, dim_3), outputPath) #Number of cells=1782 in 3 directions (x,y,z)
    
    widgets = ["Building Dataset: ", progressbar.Percentage(), " ",progressbar.Bar(), " ", progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=len(paths),widgets=widgets).start()
    
    for (i, path) in enumerate(paths):

        force, disp = Preprocessor.array_reshape(path, config.data_shape, channel_firtst=False)       
        writer.add([force], [disp])
        pbar.update(i)
    
    pbar.finish()
    writer.close()                        