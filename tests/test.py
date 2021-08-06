import os
import roboflow

from dotenv import load_dotenv

load_dotenv()

if __name__ == '__main__':
    rflow = roboflow.auth(os.getenv("ROBOFLOW_API_KEY_2"))
    print(rflow)
