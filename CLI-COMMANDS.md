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
