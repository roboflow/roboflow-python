import io
import urllib

from PIL import Image
from requests_toolbelt.multipart.encoder import MultipartEncoder
from tqdm import tqdm

from roboflow.config import API_URL
from roboflow.util import http
from roboflow.util.image_utils import validate_image_path
from roboflow.util.prediction import PredictionGroup


class InferenceModel:
    def __init__(
        self,
        api_key,
        version_id,
        colors=None,
        *args,
        **kwargs,
    ):
        """
        Create an InferenceModel object through which you can run inference.

        Args:
            api_key (str): private roboflow api key
            version_id (str): the ID of the dataset version to use for inference
        """

        self.__api_key = api_key
        self.id = version_id

        if version_id != "BASE_MODEL":
            version_info = self.id.rsplit("/")
            self.dataset_id = version_info[1]
            self.version = version_info[2]
            self.colors = {} if colors is None else colors

    def __get_image_params(self, image_path):
        """
        Get parameters about an image (i.e. dimensions) for use in an inference request.

        Args:
            image_path (Union[str, np.ndarray]): path to image or numpy array

        Returns:
            Tuple containing a dict of querystring params and a dict of requests kwargs

        Raises:
            Exception: Image path is not valid
        """
        import numpy as np

        if isinstance(image_path, np.ndarray):
            # Convert numpy array to PIL Image
            image = Image.fromarray(image_path)
            dimensions = image.size
            image_dims = {"width": str(dimensions[0]), "height": str(dimensions[1])}
            buffered = io.BytesIO()
            image.save(buffered, quality=90, format="JPEG")
            data = MultipartEncoder(fields={"file": ("imageToUpload", buffered.getvalue(), "image/jpeg")})
            return {}, {"data": data, "headers": {"Content-Type": data.content_type}}, image_dims

        validate_image_path(image_path)

        hosted_image = urllib.parse.urlparse(image_path).scheme in ("http", "https")

        if hosted_image:
            image_dims = {"width": "Undefined", "height": "Undefined"}
            return {"image": image_path}, {}, image_dims

        image = Image.open(image_path)
        dimensions = image.size
        image_dims = {"width": str(dimensions[0]), "height": str(dimensions[1])}
        buffered = io.BytesIO()
        image.save(buffered, quality=90, format="JPEG")
        data = MultipartEncoder(fields={"file": ("imageToUpload", buffered.getvalue(), "image/jpeg")})
        return (
            {},
            {"data": data, "headers": {"Content-Type": data.content_type}},
            image_dims,
        )

    def predict(self, image_path, prediction_type=None, **kwargs):
        """
        Infers detections based on image from a specified model and image path.

        Args:
            image_path (str): path to the image you'd like to perform prediction on
            prediction_type (str): type of prediction to perform
            **kwargs: Any additional kwargs will be turned into querystring params

        Returns:
            PredictionGroup Object

        Raises:
            Exception: Image path is not valid

        Example:
            >>> import roboflow

            >>> rf = roboflow.Roboflow(api_key="")

            >>> project = rf.workspace().project("PROJECT_ID")

            >>> model = project.version("1").models()[0]

            >>> prediction = model.predict("YOUR_IMAGE.jpg")
        """
        params, request_kwargs, image_dims = self.__get_image_params(image_path)

        params["api_key"] = self.__api_key

        params.update(**kwargs)
        url = f"{self.api_url}?{urllib.parse.urlencode(params)}"  # type: ignore[attr-defined]
        response = http.post(url, **request_kwargs)
        response.raise_for_status()

        return PredictionGroup.create_prediction_group(
            response.json(),
            image_path=image_path,
            prediction_type=prediction_type,
            image_dims=image_dims,
            colors=self.colors,
        )

    def download(self, format="pt", location="."):
        """
        Download the weights associated with a model.

        Args:
            format (str): The format of the output.
                        - 'pt': returns a PyTorch weights file
            location (str): The location to save the weights file to
        """
        supported_formats = ["pt"]
        if format not in supported_formats:
            raise Exception(f"Unsupported format {format}. Must be one of {supported_formats}")

        workspace, project, version = self.id.rsplit("/")

        # get pt url
        pt_api_url = f"{API_URL}/{workspace}/{project}/{self.version}/ptFile"

        r = http.get(pt_api_url, params={"api_key": self.__api_key})

        r.raise_for_status()

        pt_weights_url = r.json()["weightsUrl"]

        response = http.get(pt_weights_url, stream=True)

        # write the zip file to the desired location
        with open(location + "/weights.pt", "wb") as f:
            total_length = int(response.headers.get("content-length"))  # type: ignore[arg-type]
            for chunk in tqdm(
                response.iter_content(chunk_size=1024),
                desc=f"Downloading weights to {location}/weights.pt",
                total=int(total_length / 1024) + 1,
            ):
                if chunk:
                    f.write(chunk)
                    f.flush()

        return
