In order to use the tests located in `tests.py` you must create a `config.json` file with several 
variables for the tests to use. Your JSON should look like:
```
{
    "ROBOFLOW_KEY": "YOUR_ROBOFLOW_KEY",
    "WORKSPACE_NAME": "WORKSPACE_TO_TEST",
    "PROJECT_NAME": "PROJECT_TO_TEST",
    "IMAGE_NAME": "IMAGE_TO_UPLOAD"
}
```

Do not commit the `config.json` to the repository. 