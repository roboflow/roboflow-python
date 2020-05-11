import roboflow
roboflow.auth("<<YOUR API KEY>>")
info = roboflow.load("chess-sample", 1, "tfrecord")
