# The roboflow-python command line
This has the same capabilities of the [roboflow node cli](https://www.npmjs.com/package/roboflow-cli) so that our users don't need to install two different tools.

## See available commands

```bash
$ roboflow --help
```

```
usage: roboflow [-h] {login,download,upload,import,infer,project,workspace} ...

Welcome to the roboflow CLI: computer vision at your fingertips 🪄

options:
  -h, --help            show this help message and exit

subcommands:
  {login,download,upload,import,infer,project,workspace}
    login               Log in to Roboflow
    download            Download a dataset version from your workspace or Roboflow Universe.
    upload              Upload a single image to a dataset
    import              Import a dataset from a local folder
    infer               perform inference on an image
    project             project related commands. type 'roboflow project' to see detailed command help
    workspace           workspace related commands. type 'roboflow workspace' to see detailed command help
```

## Authentication

You need to authenticate first

```bash
$ roboflow login
```

```
visit https://app.roboflow.com/auth-cli to get your authentication token.
Paste the authentication token here:
```
Open that link on your browser, get the token, paste it on the terminal.
The credentials get saved to `~/.config/roboflow/config.json`

## Display help usage for other commands

"How do I download stuff?"

```bash
$ roboflow download --help
```
```
usage: roboflow download [-h] [-f FORMAT] [-l LOCATION] datasetUrl

positional arguments:
  datasetUrl   Dataset URL (e.g., `roboflow-100/cells-uyemf/2`)

options:
  -h, --help   show this help message and exit
  -f FORMAT    Specify the format to download the version. Available options: [coco, yolov5pytorch, yolov7pytorch, my-yolov6, darknet,
               voc, tfrecord, createml, clip, multiclass, coco-segmentation, yolo5-obb, png-mask-semantic, yolov8]
  -l LOCATION  Location to download the dataset
```

"How do I import a dataset into my workspace?"

```bash
$ roboflow import --help
```

```
usage: roboflow import [-h] [-w WORKSPACE] [-p PROJECT] [-c CONCURRENCY] [-f FORMAT] folder

positional arguments:
  folder          filesystem path to a folder that contains your dataset

options:
  -h, --help      show this help message and exit
  -w WORKSPACE    specify a workspace url or id (will use default workspace if not specified)
  -p PROJECT      project will be created if it does not exist
  -c CONCURRENCY  how many image uploads to perform concurrently (default: 10)
  -f FORMAT       dataset format. Valid options are [voc, yolov8, yolov5, auto] (use auto for autodetect)
```

## Example: download dataset

Download [Joseph's chess dataset](https://universe.roboflow.com/joseph-nelson/chess-pieces-new/dataset/25) from Roboflow Universe in VOC format:

```bash
$ roboflow download -f voc -l ~/tmp/chess joseph-nelson/chess-pieces-new/25
```
```
loading Roboflow workspace...
loading Roboflow project...
Downloading Dataset Version Zip in /Users/tony/tmp/chess to voc:: 100%|██████████████████████████| 19178/19178 [00:01<00:00, 10424.62it/s]

Extracting Dataset Version Zip to /Users/tony/tmp/chess in voc:: 100%|██████████████████████████████| 1391/1391 [00:00<00:00, 8992.30it/s]
```
```bash
$ ls -lh ~/tmp/chess
total 16
-rw-r--r--@    1 tony  staff   1.8K Jan  5 10:32 README.dataset.txt
-rw-r--r--@    1 tony  staff   562B Jan  5 10:32 README.roboflow.txt
drwxr-xr-x@   60 tony  staff   1.9K Jan  5 10:32 test
drwxr-xr-x@ 1214 tony  staff    38K Jan  5 10:32 train
drwxr-xr-x@  118 tony  staff   3.7K Jan  5 10:32 valid
```

## Example: list workspaces
List the workspaces you have access to

```bash
$ roboflow workspace list
```

```
tonyprivate
  link: https://app.roboflow.com/tonyprivate
  id: tonyprivate

wolfodorpythontests
  link: https://app.roboflow.com/wolfodorpythontests
  id: wolfodorpythontests

test minimize
  link: https://app.roboflow.com/test-minimize
  id: test-minimize
```

## Example: get workspace details

```bash
$ roboflow workspace get tonyprivate
```

```
{
  "workspace": {
    "name": "tonyprivate",
    "url": "tonyprivate",
    "members": 4,
    "projects": [
      {
        "id": "tonyprivate/annotation-upload",
        "type": "object-detection",
        "name": "annotation-upload",
        "created": 1685199749.708,
        "updated": 1695910515.48,
        "images": 1,
        (...)
      }
    ]
  }
}
```

## Example: list projects

```bash
roboflow project list -w tonyprivate
```
```
annotation-upload
  link: https://app.roboflow.com/tonyprivate/annotation-upload
  id: tonyprivate/annotation-upload
  type: object-detection
  versions: 0
  images: 1
  classes: dict_keys(['0', 'Rabbits1', 'Rabbits2', 'minion1', 'minion0', '5075E'])

hand-gestures
  link: https://app.roboflow.com/tonyprivate/hand-gestures-fsph8
  id: tonyprivate/hand-gestures-fsph8
  type: object-detection
  versions: 5
  images: 387
  classes: dict_keys(['zero', 'four', 'one', 'two', 'five', 'three', 'Guard'])
```

## Example: get project details

```bash
roboflow project get -w tonyprivate annotation-upload
```
```
{
  "workspace": {
    "name": "tonyprivate",
    "url": "tonyprivate",
    "members": 4
  },
  "project": {
    "id": "tonyprivate/annotation-upload",
    "type": "object-detection",
    "name": "annotation-upload",
    "created": 1685199749.708,
    "updated": 1695910515.48,
    "images": 1,
    (...)
  },
  "versions": []
}
```

## Example: run inference

If your project has a trained model (or you are using a dataset from Roboflow Universe that has a trained model), you can run inference from the command line.

Let's use [Rock-Paper-Scissors sample public dataset]([url](https://universe.roboflow.com/roboflow-58fyf/rock-paper-scissors-sxsw/model/11)) from Roboflow universe

(In my case, `~/scissors.png` is me holding two fingers to the camera, you can use your own image file ;-))

```bash
roboflow infer -w roboflow-58fyf -m rock-paper-scissors-sxsw/11 ~/scissors.png
```
```
{
  "x": 1230.0,
  "y": 814.5,
  "width": 840.0,
  "height": 1273.0,
  "confidence": 0.8817358016967773,
  "class": "Scissors",
  "class_id": 2,
  "image_path": "/Users/tony/scissors.png",
  "prediction_type": "ObjectDetectionModel"
}
```
