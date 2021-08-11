import os
import roboflow

from dotenv import load_dotenv

load_dotenv()

if __name__ == '__main__':
    rflow = roboflow.auth(os.getenv("ROBOFLOW_API_KEY_3"))
    rflow.list_datasets()
    project = rflow.load("cats-dogs")
    model = project.model(1)
    prediction = model.predict("https://i.imgur.com/XG2UtK7.jpg", hosted=True)
    prediction.plot()
    prediction.save(output_path='classify.jpg')
