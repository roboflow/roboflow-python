import os
import roboflow

from dotenv import load_dotenv

load_dotenv()

if __name__ == '__main__':
    # Authenticate roboflow

    rflow = roboflow.Roboflow(api_key="API_KEY")
    # List all datasets based on user
    rflow.list_datasets()
    # Load a certain project
    project = rflow.load("cats-dogs")
    # Upload image to dataset
    project.upload("https://i.imgur.com/XG2UtK7.jpg", hosted_image=True)
    # Choose a specific model from the project
    model = project.model(1)
    # predict on a certain image
    prediction = model.predict("https://i.imgur.com/XG2UtK7.jpg", hosted=True)
    # Plot the prediction
    prediction.plot()
    # Save the prediction as an image
    prediction.save(output_path='classify.jpg')
