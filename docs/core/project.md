## Action Recognition video segments

Pass the **final** `videoId` from the uploaded video status to
`Project.annotate_video_segments`. An upload's initial ID can change when the
server reuses an existing video Source. The annotation document uses the
official `roboflow-video-coco` format; keep its PTS and rational time base from
the video rather than converting them to seconds or rounding frames.

```python
import roboflow

project = roboflow.Roboflow(api_key="MY_API_KEY").workspace("my-workspace").project("actions")
video_id = final_upload_status["videoId"]  # status is "uploaded"
document = {
    "info": {"format": "roboflow-video-coco"},
    "videos": [{
        "id": 1, "file_name": "clip.mp4", "width": 640, "height": 360,
        "duration": 10, "fps": 24,
        "time_base": {"numerator": 1, "denominator": 12288},
    }],
    "categories": [{"id": 1, "name": "jumping"}],
    "segments": [{
        "id": 1, "video_id": 1, "category_id": 1,
        "start_frame": 48, "end_frame": 95,
        "start_pts": 24576, "end_pts": 49152,
    }],
}
result = project.annotate_video_segments(video_id, document)
```

The API adds the video to the Dataset by default. Use `add_to_dataset=False`
to preserve membership, `split="valid"` to choose a split, or `overwrite=True`
to replace different existing segments. An identical retry succeeds. A
conflicting annotation raises `AnnotationSaveError` with `status_code == 409`;
the server keeps the prior segments. This route requires the public video
annotate API from [platform PR #16236](https://github.com/roboflow/roboflow/pull/16236)
and its [atomic save prerequisite](https://github.com/roboflow/roboflow/pull/16456).

:::roboflow.core.project
