from utils.config import config
from utils.loader import DatasetLoader
from utils.config import config
from utils.io import HDF5DatasetGenerator
from utils.nn import UNet
from utils.monitor import TrainingMonitor
import os


print("[INFO] initialize data generator...")
trainGen = HDF5DatasetGenerator(config.TRAIN_HDF5, batchSize=32)
valGen = HDF5DatasetGenerator(config.VAL_HDF5, batchSize=32)        

print("[INFO] compiling model...")
model = UNet.build(input_size=config.data_shape)

path = os.path.sep.join([config.OUTPUT_PATH, "{}.png".format(os.getpid())])
callbacks = [TrainingMonitor(path)]

model.fit_generator(
    trainGen.generator(),
    steps_per_epoch=trainGen.numFiles // 2,
    validation_data=valGen.generator(),
    validation_steps=valGen.numFiles // 2,
    epochs=20,
    max_queue_size=10,
    callbacks=callbacks, verbose=1
    )