# Copyright 2017 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Python interface for Boxes proto.

Support read and write of Boxes from/to numpy arrays and file.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from delf import box_pb2

import os
import xml.etree.ElementTree as ET

def ArraysToBoxes(im_name, boxes, scores, class_indices):
  """Converts `boxes` to Boxes proto.

  Args:
    boxes: [N, 4] float array denoting bounding box coordinates, in format [top,
      left, bottom, right].
    scores: [N] float array with detection scores.
    class_indices: [N] int array with class indices.

  Returns:
    boxes_proto: Boxes object.
  """
  num_boxes = len(scores)
  assert num_boxes == boxes.shape[0]
  assert num_boxes == len(class_indices)

  boxes_proto = box_pb2.Boxes()
  for i in range(num_boxes):
    boxes_proto.box.add(
        ymin=boxes[i, 0],
        xmin=boxes[i, 1],
        ymax=boxes[i, 2],
        xmax=boxes[i, 3],
        score=scores[i],
        class_index=class_indices[i])

  bounding_boxes = []
  for box_proto in boxes_proto.box:
      ymin = box_proto.ymin
      xmin = box_proto.xmin
      ymax = box_proto.ymax
      xmax = box_proto.xmax
      bounding_boxes.append([ymin, xmin, ymax, xmax])

    
  folder_path = "/content/models/research/delf/delf/python/examples/data/BB_Coordinates"
  
  os.makedirs(folder_path, exist_ok=True)
  file_path = os.path.join(folder_path, "bounding_boxes.txt")
  if os.path.exists(file_path):
      mode = 'a'  # Append mode
  else:
      mode = 'w'  # Write mode (create new file)
   
  with open(file_path, mode) as file:
        # Write the bounding box coordinates
        for i, box in enumerate(boxes):
          #file.write(f"Box {i+1}: ")
          file.write(f"name: {im_name},ymin: {box[0]}, xmin: {box[1]}, ymax: {box[2]}, xmax: {box[3]}\n")


  return boxes_proto


def BoxesToArrays(boxes_proto):
  """Converts data saved in Boxes proto to numpy arrays.

  If there are no boxes, the function returns three empty arrays.

  Args:
    boxes_proto: Boxes proto object.

  Returns:
    boxes: [N, 4] float array denoting bounding box coordinates, in format [top,
      left, bottom, right].
    scores: [N] float array with detection scores.
    class_indices: [N] int array with class indices.
  """
  num_boxes = len(boxes_proto.box)
  if num_boxes == 0:
    return np.array([]), np.array([]), np.array([])

  boxes = np.zeros([num_boxes, 4])
  scores = np.zeros([num_boxes])
  class_indices = np.zeros([num_boxes])

  for i in range(num_boxes):
    box_proto = boxes_proto.box[i]
    boxes[i] = [box_proto.ymin, box_proto.xmin, box_proto.ymax, box_proto.xmax]
    scores[i] = box_proto.score
    class_indices[i] = box_proto.class_index
  
    
  return boxes, scores, class_indices


def SerializeToString(im_name, boxes, scores, class_indices):
  """Converts numpy arrays to serialized Boxes.

  Args:
    boxes: [N, 4] float array denoting bounding box coordinates, in format [top,
      left, bottom, right].
    scores: [N] float array with detection scores.
    class_indices: [N] int array with class indices.

  Returns:
    Serialized Boxes string.
  """
  boxes_proto = ArraysToBoxes(im_name, boxes, scores, class_indices)
  return boxes_proto.SerializeToString()


def ParseFromString(string):
  """Converts serialized Boxes proto string to numpy arrays.

  Args:
    string: Serialized Boxes string.

  Returns:
    boxes: [N, 4] float array denoting bounding box coordinates, in format [top,
      left, bottom, right].
    scores: [N] float array with detection scores.
    class_indices: [N] int array with class indices.
  """
  boxes_proto = box_pb2.Boxes()
  boxes_proto.ParseFromString(string)
  return BoxesToArrays(boxes_proto)


def ReadFromFile(file_path):
  """Helper function to load data from a Boxes proto format in a file.

  Args:
    file_path: Path to file containing data.

  Returns:
    boxes: [N, 4] float array denoting bounding box coordinates, in format [top,
      left, bottom, right].
    scores: [N] float array with detection scores.
    class_indices: [N] int array with class indices.
  """
  with tf.io.gfile.GFile(file_path, 'rb') as f:
    return ParseFromString(f.read())


def WriteToFile(file_path, boxes, scores, class_indices):
  """Helper function to write data to a file in Boxes proto format.

  Args:
    file_path: Path to file that will be written.
    boxes: [N, 4] float array denoting bounding box coordinates, in format [top,
      left, bottom, right].
    scores: [N] float array with detection scores.
    class_indices: [N] int array with class indices.
  """
  file_name = os.path.basename(file_path)
  # Split the file name and extension
  im_name = os.path.splitext(file_name)[0]
  serialized_data = SerializeToString(im_name, boxes, scores, class_indices)
  with tf.io.gfile.GFile(file_path, 'w') as f:
    f.write(serialized_data)
