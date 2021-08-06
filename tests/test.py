import os
import roboflow
from roboflow.models.object_detection import ObjectDetectionModel

if __name__ == '__main__':
    # Create Model with API Key + Model Endpoint
    model = ObjectDetectionModel(api_key=os.getenv('ROBOFLOW_API_KEY'), dataset_slug=os.getenv('ROBOFLOW_MODEL'),
                                 version=os.getenv('DATASET_VERSION'), stroke=3)
    # Get prediction via an image
    prediction_group = model.predict("rabbit2.jpg", hosted=False, format="json")

    # Plot predictions using matplotlib
    prediction_group.plot()
    # prediction_group.add_prediction(Prediction(
    #     {'x': 1, 'y': 1, 'width': 1, 'height': 1, 'class': 'hi', 'confidence': 0.1}, 'dskfj.jpg'))
