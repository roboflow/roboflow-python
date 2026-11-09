# Upload a custom OWLv2 model

Use `model_type="owlv2"` to upload fine-tuned OWLv2 weights for Auto Label and Workflows.
For a standard OWLv2 model, the only required file is the checkpoint:

```text
weights-upload/
  owlv2.pth
```

`owlv2.pth` must contain the complete `Owlv2ForObjectDetection.state_dict()`, including the text encoder and detection heads.
A training checkpoint that wraps the weights under another key is not accepted.
To use image-only requests in Auto Label or the Workflows Object Detection Model block, add `class_names.txt` beside the checkpoint.
It contains unique, nonblank output labels, one per line. Class IDs follow this order. By default, these labels are also the text queries.
If you upload without `class_names.txt`, use `workspace.deploy_model`. An upload to a dataset version requires the labels file.

The exporter assumes the standard `google/owlv2-base-patch16-ensemble` model configuration, image preprocessing, and tokenizer.
It bundles these defaults at revision `cfd3195ba4ea9592eec887ded089f4c08eff231d` and uses them offline.
Fine-tuning the weights does not require additional files if those components stay unchanged.
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

## Run local inference

On a local server that supports `/v2/models/infer`, send your model ID, image, and text queries.
Replace the model ID and image path below. The example uses the same API key as the upload.

```bash
curl 'http://localhost:9001/v2/models/infer?model_id=your-workspace/custom-owlv2' \
  -H "Authorization: Bearer $ROBOFLOW_API_KEY" \
  -F 'image=@field.jpg' \
  -F 'inputs={"classes":["maize tassel","weed"]}'
```

Each class ID indexes the query list for that request. Explicit queries override any saved prompts for that call.
If you included `class_names.txt`, omit the `inputs` field to use the saved labels, or select this model in Auto Label.
If you uploaded only the weights, supply queries on every request.

## Custom model or preprocessing

If training changed a component, place its override files beside the weights.
Each supplied component takes precedence over its default.

| Optional files | When to supply them |
| --- | --- |
| `config.json` | A different model architecture, saved with `model.config.save_pretrained(folder)` |
| `processor_config.json` or `preprocessor_config.json` | Different image preprocessing, such as resizing or normalization |
| Tokenizer files | Different tokenization, saved with the original tokenizer or processor |
| `owlv2_prompts.json` | Text prompts that differ from the output labels |

A tokenizer override needs `tokenizer_config.json` and either `tokenizer.json` or both `vocab.json` and `merges.txt`.
Include additional tokenizer files that the original processor saved, such as `added_tokens.json` and `special_tokens_map.json`.
Partial tokenizer overrides produce an error. They are never mixed with the default tokenizer.
You can save a complete processor override with `processor.save_pretrained(folder)`.

`owlv2_prompts.json` is a JSON array with one nonblank prompt per label, in label order.
For example, labels `dent` and `scratch` can use `["a dent in a metal panel", "a scratch in a metal panel"]`.

The exporter strictly matches the weights to the selected model configuration and rejects partial or adapter-only checkpoints.
If the weights do not match the default architecture, supply the original `config.json`.
Weight shapes cannot identify custom resizing or normalization. Omitted image preprocessing files mean standard preprocessing.
The SDK packages files without loading the checkpoint. Before the upload, it checks only that the set of files is consistent.
The exporter validates the contents of the files.
