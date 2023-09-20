import io
import urllib

import requests
from PIL import Image
from requests_toolbelt.multipart.encoder import MultipartEncoder

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

        version_info = self.id.rsplit("/")
        self.dataset_id = version_info[1]
        self.version = version_info[2]
        self.colors = {} if colors is None else colors

    def __get_image_params(self, image_path):
        """
        Get parameters about an image (i.e. dimensions) for use in an inference request.

        Args:
            image_path (str): path to the image you'd like to perform prediction on

        Returns:
            Tuple containing a dict of querystring params and a dict of requests kwargs

        Raises:
            Exception: Image path is not valid
        """
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
        data = MultipartEncoder(
            fields={"file": ("imageToUpload", buffered.getvalue(), "image/jpeg")}
        )
        return (
            {},
            {"data": data, "headers": {"Content-Type": data.content_type}},
            image_dims,
        )

    def __failsafe_post(
        self, url, data=None, headers=None, timeout=-1, max_retries=1, **kwargs
    ):
        """
        Send a POST request with defined timeout and maximum number of retries

        Args:
            url (str): URL to POST to
            data (dict): payload to send in request
            headers (dict): headers to send in request
            timeout (int): maximum time to wait for a response
            max_retries (int): maximum number of times to retry any failed requests

        Returns:
            resp (dict): response from request
        """
        if timeout == -1:
            resp = requests.post(url=url, data=data, headers=headers, **kwargs)
        else:
            while max_retries > 0:
                try:
                    resp = requests.post(
                        url=url, data=data, headers=headers, timeout=timeout, **kwargs
                    )
                    break
                except:
                    max_retries -= 1

        return resp

    def predict(
        self, image_path, prediction_type=None, timeout=-1, max_retries=1, **kwargs
    ):
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

            >>> model = project.version("1").model

            >>> prediction = model.predict("YOUR_IMAGE.jpg")
        """
        params, request_kwargs, image_dims = self.__get_image_params(image_path)

        params["api_key"] = self.__api_key

        params.update(**kwargs)

        url = f"{self.api_url}?{urllib.parse.urlencode(params)}"
        response = self.__failsafe_post(
            url=url, timeout=timeout, max_retries=max_retries, **request_kwargs
        )
        response.raise_for_status()

        return PredictionGroup.create_prediction_group(
            response.json(),
            image_path=image_path,
            prediction_type=prediction_type,
            image_dims=image_dims,
            colors=self.colors,
        )
