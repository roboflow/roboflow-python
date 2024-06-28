import roboflow

rf = roboflow.Roboflow(api_key="HgKBrG6GzhfssPbg0mEA")

workspace = rf.workspace("leo-ueno")

yes_model = workspace.project("people-detection-o4rdr").version(7).model
print("There is a model:", type(yes_model))

no_model = workspace.project("people-detection-o4rdr").version(8).model
print(
    "There is no model:", type(no_model)
)  # Previously would have returned a <class 'roboflow.models.object_detection.ObjectDetectionModel'>
