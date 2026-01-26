import os, glob
import constant

train_folder = constant.TRAIN_FOLDER
batch_size = constant.TRAIN_BATCH_SIZE
iterations = constant.ITERATIONS
summary_dir = constant.SUMMARY_DIR
snapshot_dir = constant.SNAPSHOT_DIR

print("CWD =", os.getcwd())
print("TRAIN_FOLDER =", os.path.abspath(train_folder))
print("TRAIN_FOLDER exists =", os.path.isdir(train_folder))
print("TRAIN_FOLDER files sample =", glob.glob(os.path.join(train_folder, "*"))[:20])
