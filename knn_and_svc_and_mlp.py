# How to run: python3 knn_and_svc_and_mlp.py --dataset ./train

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from preprocessing.preprocessor import Preprocessor
from datasets.datasetloader import DatasetLoader
from imutils import paths
import argparse

ap = argparse.ArgumentParser()

ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-k", "--neighbors", type=int, default=1, help="# of nearest neighbors for classification")
ap.add_argument("-j", "--jobs", type=int, default=-1, help="# of jobs for k-NN distance (-1 uses all available cores)")

args = vars(ap.parse_args())

print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))

sp = Preprocessor(32, 32)
sdl = DatasetLoader(preprocessors=[sp])
(data, labels) = sdl.load(imagePaths, verbose=1000)
data = data.reshape((data.shape[0], 3072))

print("[INFO] features matrix: {:.1f}MB".format(data.nbytes / (1024 * 1000.0)))

le = LabelEncoder()
labels = le.fit_transform(labels)
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

#k-NN
# print("[INFO] evaluating k-NN classifier...")
# model = KNeighborsClassifier(n_neighbors=args["neighbors"], n_jobs=args["jobs"])
# model.fit(trainX, trainY)
# print(classification_report(testY, model.predict(testX), target_names=le.classes_))


#SVM
print("\n")
print("[INFO] evaluating SVM classifier...")
model = SVC(max_iter=1000,class_weight='balanced')
model.fit(trainX, trainY)
acc = model.score(testX, testY)
print("[INFO] SVM-SVC raw pixel accuracy: {:.2f}%".format(acc * 100))
