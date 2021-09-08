import os
import roboflow

from dotenv import load_dotenv

load_dotenv()

if __name__ == '__main__':
    #Authenticate roboflow
    rflow = roboflow.Roboflow(api_key="API_KEY")
    #Retrieve specific project
    project = rflow.project("PROJECT_NAME")
    #List all versions
    project.versions()
    #Get first version
    project.version("1")
    #Retrieve first model version
    model = project.version("1").model
    #Predict on a certain image
    prediction = model.predict("IMAGE_NAME")
    #Plot the prediction
    prediction.plot()
    #Save the prediction as an image
    prediction.save(output_path='classify.jpg')
