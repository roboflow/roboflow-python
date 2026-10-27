# Upload a custom OWLv2 model

Use `model_type="owlv2-custom"` to upload a complete OWLv2 parameter dictionary.
This model uses saved text prompts and returns object detections with fixed labels.
It works through the custom-model Auto Label path and the Object Detection Model Workflow block.

## Prepare the upload folder

Save these files beside the checkpoint:

| File | Content |
| --- | --- |
| `owlv2.pth` | The complete `Owlv2ForObjectDetection.state_dict()`, including the text encoder and detection heads |
| `config.json` | The original model configuration, saved with `model.config.save_pretrained(folder)` |
| Processor and tokenizer files | The original processor, saved with `processor.save_pretrained(folder)` |
| `class_names.txt` | Unique output labels, one per line |
| `owlv2_prompts.json` | Optional JSON array of text prompts, in label order |

The processor files must include `tokenizer_config.json` and either `tokenizer.json` or both `vocab.json` and `merges.txt`.
Include either `processor_config.json` or `preprocessor_config.json` for image processing.

If you omit `owlv2_prompts.json`, the model uses each label as its text prompt.
For example, labels `dent` and `scratch` can use prompts `["a dent in a metal panel", "a scratch in a metal panel"]`.
The class IDs follow the order in `class_names.txt`.

The checkpoint alone does not identify the original preprocessing or text prompts.
Obtain those values from the training or inference code.
The exporter rejects partial checkpoints and adapter-only weights.
The SDK packages files without loading the checkpoint.

## Upload

Use an object-detection project that the workspace can access.
Replace the workspace, project, folder, and model name with your values.
Set `ROBOFLOW_API_KEY` in your environment before you run this example.

```python
from roboflow import Roboflow

workspace = Roboflow().workspace("your-workspace")
workspace.deploy_model(
    model_type="owlv2-custom",
    model_path="/absolute/path/to/weights-upload",
    filename="owlv2.pth",
    project_ids=["your-project"],
    model_name="custom-owlv2",
)
```

After upload processing finishes, select the uploaded model under custom models in Auto Label.
In Workflows, select its model ID in an Object Detection Model block.
The saved prompts stay fixed for each model package.

This feature requires the corresponding platform, rflm exporter, and inference runtime releases.
