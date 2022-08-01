import json
from time import sleep
from PIL import Image
import io
import base64
import requests
from os.path import exists

# a for loop that counts the number of occurances within an array
def count_object_occurances(predictions, target_class):
  """
    Helper method to count the number of objects in an image for a given class
    :param predictions: predictions returned from calling the predict method
    :param target_class: str, target class for object count
    :return: dictionary with target class and total count of occurrences in image
  """
  object_counts = {target_class : 0}
  for prediction in predictions:
    if prediction['class'] in target_class:
      object_counts[prediction['class']] += 1
  return object_counts

# compares counts and returns False if counts below requirement
def object_count_comparisons(predictions, required_objects_count, required_class_count, target_class):
  """
    Helper method to count the number of objects in an image for a given class
    and return whether or not the class was found and the number of occurrences
    :param predictions: predictions returned from calling the predict method
    :param required_objects_count: integer, number of occurrences for located
    objects
    :param required_class_count: integer, number of class occurrences for located
    object
    :param target_class: str, target class for object count
    :return: dictionary with target class and total count of occurrences in image
  """
  if(len(predictions) < required_objects_count or 
    target_class and
    count_object_occurances(predictions, target_class)[target_class] < required_class_count):  
      return f"""There are fewer than {required_objects_count} objects or {required_class_count}
      occurrences of the '{target_class}' class present.)"""
  else:
    occurances = count_object_occurances(predictions, target_class)
    return occurances
