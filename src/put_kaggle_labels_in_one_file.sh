#/bin/bash

head -1 set_a.csv > labels_kaggle.csv
sed 1d set_a.csv >> labels_kaggle.csv
sed 1d set_b.csv >> labels_kaggle.csv
