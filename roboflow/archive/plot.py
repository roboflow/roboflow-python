# import io
# import os
#
# import numpy as np
# import requests
# from PIL import Image
# import matplotlib.pyplot as plt
# from matplotlib import patches
#
# from roboflow.util.image_utils import check_image_url
# from roboflow.config import OBJECT_DETECTION_MODEL, CLASSIFICATION_MODEL
#
#
# def __plot_image(image_path):
#     """
#     Helper method to plot image
#
#     :param image_path: path of image to be plotted (can be hosted or local)
#     :return:
#     """
#     # Exception to check if image path exists
#     __exception_check(image_path_check=image_path)
#     # Try opening local image
#     try:
#         img = Image.open(image_path)
#     except OSError:
#         # Try opening Hosted image
#         response = requests.get(image_path)
#         img = Image.open(io.BytesIO(response.content))
#     # Plot image axes
#     figure, axes = plt.subplots()
#     axes.imshow(img)
#     return figure, axes
#
#
# def __plot_annotation(axes, prediction=None, stroke=1):
#     """
#     Helper method to plot annotations
#
#     :param axes:
#     :param prediction:
#     :return:
#     """
#     # Object Detection annotation
#     if prediction['prediction_type'] == OBJECT_DETECTION_MODEL:
#         # Get height, width, and center coordinates of prediction
#         if prediction is not None:
#             height = prediction['height']
#             width = prediction['width']
#             x = prediction['x']
#             y = prediction['y']
#             rect = patches.Rectangle((x - width / 2, y - height / 2), width, height,
#                                      linewidth=stroke, edgecolor='r', facecolor='none')
#             # Plot Rectangle
#             axes.add_patch(rect)
#     elif prediction['prediction_type'] == CLASSIFICATION_MODEL:
#         axes.set_title('Class: ' + prediction['top'] + " | Confidence: " + prediction['confidence'])
#
#
# def plot_predictions(prediction=None, prediction_group=None, binary_data=None, stroke=1):
#     """
#
#     :param image_path:
#     :param prediction:
#     :param prediction_group:
#     :return:
#     """
#     # Check if user has inputted prediction
#     if prediction is not None:
#         # Exception to check if image path exists
#         __exception_check(image_path_check=prediction['image_path'])
#         figure, axes = __plot_image(prediction['image_path'])
#
#         __plot_annotation(axes, prediction, stroke)
#         plt.show()
#     # Check if user inputted prediction group
#     elif prediction_group is not None:
#         if len(prediction_group) > 0:
#             # Check if image path exists
#             __exception_check(image_path_check=prediction_group.base_image_path)
#             # Plot image if image path exists
#             figure, axes = __plot_image(prediction_group.base_image_path)
#             # Plot annotations in prediction group
#             for single_prediction in prediction_group:
#                 __plot_annotation(axes, single_prediction, stroke)
#
#         plt.show()
#     # Raise exception if no prediction or prediction group was specified
#     elif binary_data is not None:
#         image_stream = io.BytesIO(binary_data)
#         pil_image = Image.open(image_stream)
#         plt.imshow(np.asarray(pil_image))
#         plt.show()
#     else:
#         raise Exception("No Prediction, Prediction Group, or Binary Data Specified")
#
#
# def __exception_check(image_path_check=None):
#     # Check if Image path exists exception check (for both hosted URL and local image)
#     if image_path_check is not None:
#         if not os.path.exists(image_path_check) and not check_image_url(image_path_check):
#             raise Exception("Image does not exist at " + image_path_check + "!")
