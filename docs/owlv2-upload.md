# Upload a custom OWLv2 model

After you fine-tune OWLv2 in your own environment, upload the weights to Roboflow. You supply text prompts at inference time.

## Save the checkpoint

Upload the full `Owlv2ForObjectDetection.state_dict()`. If you used `save_pretrained`, load the fine-tuned folder with `Owlv2ForObjectDetection.from_pretrained` first.

```python
from pathlib import Path
import torch

upload_dir = Path("weights-upload")
upload_dir.mkdir(exist_ok=True)
torch.save(model.state_dict(), upload_dir / "owlv2.pth")
```

The exporter assumes the standard `google/owlv2-base-patch16-ensemble` model configuration, image preprocessing, and tokenizer.
It bundles these defaults at revision `cfd3195ba4ea9592eec887ded089f4c08eff231d` and uses them offline.
If your model uses these defaults, upload only `weights-upload/owlv2.pth`.
The exporter saves the resolved configuration and processor in the inference package.

## Upload

Use an object-detection project that the workspace can access.
Replace the workspace, project, folder, and model name with your values.
Set `ROBOFLOW_API_KEY` in your environment before you run this example.

```python
from roboflow import Roboflow

workspace = Roboflow().workspace("your-workspace")
workspace.deploy_model(
    model_type="owlv2",
    model_path="/absolute/path/to/weights-upload",
    filename="owlv2.pth",
    project_ids=["your-project"],
    model_name="custom-owlv2",
)
```

In AutoLabel, select the uploaded workspace model and enter the project class names and optional visual descriptions. Preview the predictions, then start the labeling job.
Each description is one text prompt. A prompt can have at most 16 tokens, which includes the start and end tokens. A longer prompt returns an error that names the prompt.

## Use the model in a Workflow

Add the **OWLv2 Model** block (`roboflow_core/owlv2_model@v1`). Set `model_id` to `your-workspace/custom-owlv2` and `class_names` to your text prompts.
The block returns bounding boxes. To get masks, connect its `predictions` output to a Segment Anything block.

## Run local inference

On a local server that supports `/v2/models/infer`, send your model ID, image, and text queries.
Replace the model ID and image path below. The example uses the same API key as the upload.

```bash
curl 'http://localhost:9001/v2/models/infer?model_id=your-workspace/custom-owlv2' \
  -H "Authorization: Bearer $ROBOFLOW_API_KEY" \
  -F 'image=@street.jpg' \
  -F 'inputs={"classes":["bicycle","bus"]}'
```

Each class ID indexes the query list for that request. You can change the queries between images.

## Optional OWLv2 configuration files

If your model uses a different configuration, image processor, or tokenizer, place its files beside the weights.
Each supplied component takes precedence over its default.

| Optional files | When to supply them |
| --- | --- |
| `config.json` | Your `Owlv2ForObjectDetection` uses a non-default configuration, such as a different vision patch size. Save it with `model.config.save_pretrained(folder)`. |
| `processor_config.json` or `preprocessor_config.json` | Different image preprocessing, such as resizing or normalization |
| Tokenizer files | Different tokenization, saved with the original tokenizer or processor |

A tokenizer override needs `tokenizer_config.json` and either `tokenizer.json` or both `vocab.json` and `merges.txt`.
Include additional tokenizer files that the original processor saved, such as `added_tokens.json` and `special_tokens_map.json`.
Partial tokenizer overrides produce an error. They are never mixed with the default tokenizer.
You can save a complete processor override with `processor.save_pretrained(folder)`.

Weight shapes cannot identify custom resizing or normalization. Omitted image preprocessing files mean standard preprocessing.
The image preprocessing size must equal the `image_size` in the model configuration. If the sizes are different, the export fails with an error.
The SDK packages files without loading the checkpoint. Before the upload, it checks only that the set of files is consistent.
The exporter validates the contents of the files.
