from keras.utils import np_utils
import numpy as np
import h5py

class HDF5DatasetGenerator:
    
    def __init__(self, dbPath, batchSize): 
                 
        self.batchSize = batchSize
        self.db = h5py.File(dbPath)
        self.numFiles = self.db["computed_displacements"].shape[0]
        
    def generator(self):

        while True:
            for i in np.arange(0, self.numFiles, self.batchSize):

                force = self.db["external_forces"][i: i + self.batchSize]
                displacement = self.db["computed_displacements"][i: i + self.batchSize]
                    
                yield (force, displacement)
            
    def close(self):
        self.db.close()
        
        
        
        
        
        
        