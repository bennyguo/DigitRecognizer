# Digit Recognizer
<a href="https://www.kaggle.com/c/digit-recognizer">Digit recognizer competition</a> in kaggle.

# Requirements
1. Python 2.7.
2. To run cnn.py, you need tensorflow 1.0+ installed.
3. To run knn.py, you need scikit-learn package installed.
4. To run xgb.py, you need xgboost and xgboost python package installed.

# Usage
1. Download dataset from <a href="https://www.kaggle.com/c/digit-recognizer/data">kaggle website</a> and put train.csv and test.csv in root path of this project.
2. Run cnn.py or knn.py or xgb.py in python.
3. Collect logs and submissions in logs/ folder and submissions/ folder.

# Options
## General options
- **-n \<name\>** [Model name.]
- **--expand** [Use expanded data.]
## cnn.py
- **-e \<epoches\>** [Training epoches.]
## knn.py
- **-k \<neighbors\>** [Nearest neighbor numbers.]
## xgb.py
None provided.

