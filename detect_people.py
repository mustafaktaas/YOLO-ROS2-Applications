#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
import numpy as np
import tf_transformations
import pandas
import cv2
import matplotlib.pyplot as plt
import torch
import tf2_ros

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, PointCloud2
from vision_msgs.msg import Detection2DArray, Detection2D
from geometry_msgs.msg import TransformStamped

from deep_sort.detection import Detection
from deep_sort import nn_matching
from deep_sort.tracker import Tracker
from deep_sort import generate_detections
from deep_sort import preprocessing as prep

from diffamr2_msgs.msg import TrackerObject, TrackerObjectArray
import pointcloud2 as pc2

class Detector(Node):

    def __init__(self):
        super().__init__('detector')
        self.deep_weights = '/home/diffamrhumble/ros2_ws/src/detect_people/deep_sort/model_data/mars-small128.pb'
        self.yolo_weights = '/home/diffamrhumble/ros2_ws/src/detect_people/detect_people/config/yolov5s.pt'
        self.yolov5 = '/home/diffamrhumble/ros2_ws/src/detect_people/detect_people/config/yolov5'
        image_topic = self.declare_parameter('image_topic', '/d435i/color/image_raw').get_parameter_value().string_value
        point_cloud_topic = self.declare_parameter('point_cloud_topic', '/d435i/depth/color/points/voxel_grid').get_parameter_value().string_value

        self._global_frame = 'base_link'
        self._frame = 'd435i_link'
        self._tf_prefix = self.declare_parameter('tf_prefix', self.get_name()).get_parameter_value().string_value

        self.id = 0
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        self._bridge = CvBridge()

        self._image_sub = self.create_subscription(Image, image_topic, self.image_callback, 30)

        if point_cloud_topic is not None:
            self._pc_sub = self.create_subscription(PointCloud2, point_cloud_topic, self.pc_callback, 30)
        else:
            self.get_logger().info('No point cloud information available. Objects will not be placed in the scene.')

        self._current_image = None
        self._current_pc = None

        self._image_pub = self.create_publisher(Image, '~/labeled_detect', 10)

        self._tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.get_logger().info('Ready to detect!')

    def image_callback(self, image):
        self.get_logger().info('girdikamera')
        self._current_image = image

    def pc_callback(self, pc):
        self.get_logger().info('girdipoint')
        self._current_pc = pc

    def run(self):
        while rclpy.ok():
            rclpy.spin_once(self)
            if self._current_image is not None:
                try:
                    if self._global_frame is not None:
                        trans = self._tf_buffer.lookup_transform(self._global_frame, self._frame, rclpy.time.Time())

                    scene = self._bridge.imgmsg_to_cv2(self._current_image, 'rgb8')

                    model = torch.hub.load(self.yolov5, 'custom', path=self.yolo_weights, source='local')
                    model.classes = [0]
                    model.eval()
                    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
                    model.to(device)

                    results = model(scene, size=640)
                    deteccao = results.pandas().xyxy[0]

                    marked = results.render()
                    marked_image = np.squeeze(marked)
                    self._image_pub.publish(self._bridge.cv2_to_imgmsg(marked_image, 'rgb8'))

                    detect = []
                    scores = []

                    for i in range(len(deteccao)):
                        detect.append(np.array([deteccao['xmin'][i], deteccao['ymin'][i], deteccao['xmax'][i] - deteccao['xmin'][i], deteccao['ymax'][i] - deteccao['ymin'][i]]))
                        scores.append(np.array(deteccao['confidence'][i]))

                    metric = nn_matching.NearestNeighborDistanceMetric("cosine", 0.5, 100)
                    tracker = Tracker(metric, max_iou_distance=0.7, max_age=70, n_init=3)
                    model_filename = self.deep_weights
                    encoder = generate_detections.create_box_encoder(model_filename)

                    features = encoder(scene, detect)
                    detections_new = [Detection(bbox, score, feature) for bbox, score, feature in zip(detect, scores, features)]
                    boxes = np.array([d.tlwh for d in detections_new])
                    scores_new = np.array([d.confidence for d in detections_new])
                    indices = prep.non_max_suppression(boxes, 1.0, scores_new)
                    detections_new = [detections_new[i] for i in indices]
                    tracker.predict()
                    tracker.update(detections_new)

                    publishers = {}
                    for track in tracker.tracks:
                        if track.is_confirmed() and track.time_since_update > 1:
                            continue
                        bbox = track.to_tlbr()
                        ymin, xmin, ymax, xmax = bbox[0], bbox[1], bbox[2], bbox[3]
                        id = track.track_id
                        publishers[id] = self.create_publisher(TrackerObjectArray, '~/person_' + str(id), 10)

                        publish_tf = False
                        if self._current_pc is None:
                            self.get_logger().info('No point cloud information available to track current object in scene')

                        else:
                            y_center = round(ymax - ((ymax - ymin) / 2))
                            x_center = round(xmax - ((xmax - xmin) / 2))
                            pc_list = list(pc2.read_points(self._current_pc, skip_nans=True, field_names=('x', 'y', 'z'), uvs=[float(x_center), float(y_center)]))

                            if len(pc_list) > 0:
                                publish_tf = True
                                tf_id = 'Person' + '_' + str(id)

                                if self._tf_prefix is not None:
                                    tf_id = self._tf_prefix + '/' + tf_id

                                tf_id = tf_id

                                point_x, point_y, point_z = pc_list[0]

                        if publish_tf:
                            object_tf = np.array([point_z, -point_x, -point_y]) # object_tf'yi numpy dizisine dönüştür
                            frame = self._frame

                            if self._global_frame is not None:
                                trans_np = np.array([trans.transform.translation.x, trans.transform.translation.y, trans.transform.translation.z]) # trans'i numpy dizisine dönüştür
                                object_tf = trans_np + object_tf
                                frame = self._global_frame

                            if object_tf is not None:
                                t = TransformStamped()
                                t.header.stamp = self.get_clock().now().to_msg()
                                t.header.frame_id = frame
                                t.child_frame_id = tf_id
                                t.transform.translation.x = object_tf[0]
                                t.transform.translation.y = object_tf[1]
                                t.transform.translation.z = object_tf[2]
                                t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = tf_transformations.quaternion_from_euler(0, 0, 0)


                                self._tf_broadcaster.sendTransform(t)

                    # Show the image with detections
                    cv2.imshow("Detections", marked_image)
                    cv2.waitKey(1)  # 1 ms delay for the OpenCV window to update

                except CvBridgeError as e:
                    self.get_logger().error(str(e))
                except tf2_ros.TransformException as e:
                    self.get_logger().error(str(e))


def main(args=None):
    rclpy.init(args=args)

    detector = Detector()

    try:
        detector.run()
    except KeyboardInterrupt:
        detector.get_logger().info('Shutting down')
    finally:
        detector.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
