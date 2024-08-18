class UploadImageError(Exception):
    """
    Exception raised for errors that occur during the image upload process.

    Attributes:
        message (str): A description of the error.
        retry_attempts (int): The number of retry attempts made before the error occurred.
    """

    def __init__(
        self,
        message="An error occurred during the image upload process.",
        retry_attempts=0,
    ):
        self.message = message
        self.retry_attempts = retry_attempts
        super().__init__(self.message)


class UploadAnnotationError(Exception):
    """
    Exception raised for errors that occur during the annotation upload process.

    Attributes:
        message (str): A description of the error.
        image_id (Optional[str]): The ID of the image associated with the error.
        image_upload_time (Optional[datetime]): The timestamp when the image upload was attempted.
        image_retry_attempts (Optional[int]): The number of retry attempts made for the image upload.
    """

    def __init__(
        self,
        message="An error occurred during the annotation upload process.",
        image_id=None,
        image_upload_time=None,
        image_retry_attempts=None,
    ):
        self.message = message
        self.image_id = image_id
        self.image_upload_time = image_upload_time
        self.image_retry_attempts = image_retry_attempts
        super().__init__(self.message)
