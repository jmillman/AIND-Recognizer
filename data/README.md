## American Sign Language Data
The data in this directory contained in `hands_condensed.csv` and `speaker.csv` has been derived from the [RWTH-BOSTON-104 Database](http://www-i6.informatik.rwth-aachen.de/~dreuw/database-rwth-boston-104.php) The hand positions are pulled directly from the database [boston104.handpositions.rybach-forster-dreuw-2009-09-25.full.xml](boston104.handpositions.rybach-forster-dreuw-2009-09-25.full.xml). 

The videos are sentences with translations provided in the database.  For purposes of this project, the sentences have been segmented into words based on slow motion examination of the files.  These segments are provided in the `test_words.csv` and `train_words.csv` files in the form of start and end frames (inclusive).  Training and Test word files have been divided as they are in the database, which is based on sentence Train and Test divisions.  The hand positions file has not been divided and contains all frame information.


Run the notebook with:
jupyter notebook asl_recognizer.ipynb