import time
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import rclpy
from filterpy.kalman import KalmanFilter
from geometry_msgs.msg import Point
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_system_default
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from visualization_msgs.msg import Marker, MarkerArray
from yolov8_msgs.msg import Detection, DetectionArray, KeyPoint3D

@dataclass
class TrackedPose():
    keypoints: Dict[int, KalmanFilter] = field(default_factory=dict)
    timestamps: Dict[int, float] = field(default_factory=dict)


class KeypointFusion(Node):
    def __init__(self):
        super().__init__('keypoint_fusion')
        self.camera1_detection_sub = self.create_subscription(DetectionArray,
                                                              '/yolo1/detections_3d',
                                                              self.process_observation,
                                                              qos_profile_system_default
                                                              )
        self.camera2_detection_sub = self.create_subscription(DetectionArray,
                                                              '/yolo2/detections_3d',
                                                              self.process_observation,
                                                              qos_profile_system_default)
        self.position_estimation_pub = self.create_publisher(MarkerArray,
                                                             '/keypoint_fusion/positions_3d',
                                                             qos_profile_system_default)
        self.segments_pub = self.create_publisher(MarkerArray,
                                                  '/keypoint_fusion/segments',
                                                  qos_profile_system_default)
        self.position_estimation_pub_timer = self.create_timer(0.05, self.publish_position_estimation)
        self.known_ids: List[int] = []
        self.poses: List[TrackedPose] = []
        self.assignment_distance_threshold_m = 0.5  # Disregard any assignment between a new observation and a tracked pose if the distance is above this threshold
        self.keypoint_timeout_s = 2.0
        self.frame_id: str = None
        self.SEG_PAIRS = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8],
                          [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
        
    def process_observation(self, msg: DetectionArray):
        """TODO"""
        valid_detections = [det for det in msg.detections if det.keypoints3d.data]
        self.assign_detections(valid_detections)
        self.clear_stale_points()


    def assign_detections(self, detections: List[Detection]) -> None:
        """Solves the linear sum assignment problem to match detections to the closest TrackedPose instance where the cost matrix is the euclidean distance between the mean XYZ position of each Detection-TrackedPose pair. Disregards any assignments where the cost (distance) is greater than assignment_distance_threshold_m"""
        
        # Assignment
        if self.poses and detections:
            detection_mean_positions = np.array(
                [np.mean([[kp.point.x, kp.point.y, kp.point.z] for kp in det.keypoints3d.data], axis=0) 
                for det in detections])
            pose_mean_positions = np.array(
                [np.mean([kf.x[:3] for kf in pose.keypoints.values()], axis=0) for pose in self.poses])
            cost_matrix = cdist(detection_mean_positions, pose_mean_positions, metric='euclidean')
            d_indices, p_indices = linear_sum_assignment(cost_matrix)
        else:
            d_indices, p_indices = [], []
            cost_matrix=None

        # Create/update tracked poses
        for d_idx, det in enumerate(detections):
            if not self.frame_id:
                self.frame_id = det.keypoints3d.frame_id
            elif det.keypoints3d.frame_id != self.frame_id:
                self.get_logger().warn(
                    f"Keypoint frame_id '{det.keypoints3d.frame_id}' does not match expected value '{self.frame_id}'")

            try:
                p_idx = p_indices[list(d_indices).index(d_idx)]
            except ValueError:
                p_idx = None
            
            if d_idx in d_indices and cost_matrix[d_idx, p_idx] < self.assignment_distance_threshold_m:
                self.update_tracked_pose(self.poses[p_idx], det)
            else:
                if cost_matrix is not None and cost_matrix[d_idx, p_idx] < self.assignment_distance_threshold_m:
                    self.get_logger().warn('Assignment found, but was over assignment_distance_threshold_m')
                self.create_tracked_pose(det)

    def create_tracked_pose(self, det: Detection) -> None:
        """Create a new TrackedPose object from a detection which could not be assigned to any prior poses"""
        pose = TrackedPose()
        kp: KeyPoint3D
        for kp in det.keypoints3d.data:
            pose.keypoints[kp.id] = self.create_kf(kp)
            pose.timestamps[kp.id] = time.time()
        self.poses.append(pose)

    def update_tracked_pose(self, pose: TrackedPose, det: Detection) -> None:
        """Update a TrackedPose with the position from a new detection"""
        kp: KeyPoint3D
        for kp in det.keypoints3d.data:
            if kp.id not in pose.keypoints.keys():
                pose.keypoints[kp.id] = self.create_kf(kp)
            else:
                pose.keypoints[kp.id].predict()
                pose.keypoints[kp.id].update([kp.point.x, kp.point.y, kp.point.z])
            pose.timestamps[kp.id] = time.time()

    def clear_stale_points(self) -> None:
        """Remove any points from all tracked poses that are older than keypoint_timeout_s. If a pose has no remaining keypoints then it is also removed"""
        stale_poses = []
        for n, pose in enumerate(self.poses):
            stale_keypoints = [m for m, t in pose.timestamps.items() if t < time.time() - self.keypoint_timeout_s]
            [(pose.keypoints.pop(m), pose.timestamps.pop(m)) for m in stale_keypoints]
            if not pose.keypoints:
                stale_poses.append(n)
        [self.poses.pop(m) for m in stale_poses]
                   

    def create_kf(self, kp: KeyPoint3D) -> KalmanFilter:
        """Create a new Kalman filter to track a keypoint instance"""
        kf = KalmanFilter(dim_x=6, dim_z=3)
        kf.x = np.array([kp.point.x, kp.point.y, kp.point.z, 0, 0, 0])  # Initial state: [x, y, z, vx, vy, vz]
        kf.P = np.diag([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])  # Initial covariance
        dt = 1.0
        kf.F = np.array([[1.0, 0.0, 0.0, dt, 0.0, 0.0],
                         [0.0, 1.0, 0.0, 0.0, dt, 0.0],
                         [0.0, 0.0, 1.0, 0.0, 0.0, dt],
                         [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                         [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])  # State transition matrix
        kf.H = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                         [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])  # Measurement matrix
        kf.Q = np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])  # Process noise covariance
        kf.R = np.diag([50, 50, 50])  # Measurement noise covariance
        return kf

    def publish_position_estimation(self) -> None:
        """Publish the estimated position of each tracked keypoint"""
        if not self.poses:
            return

        kp_marker_array = MarkerArray()
        seg_marker_array = MarkerArray()

        for n, pose in enumerate(self.poses):
            kp_marker = Marker()
            kp_marker.id = n
            kp_marker.header.frame_id = self.frame_id
            kp_marker.header.stamp = self.get_clock().now().to_msg()
            kp_marker.type = 7  # Sphere list
            kp_marker.lifetime = Duration(seconds=1).to_msg()
            kp_marker.color.r, kp_marker.color.g, kp_marker.color.b, kp_marker.color.a = 0.0, 1.0, 1.0, 0.6
            kp_marker.scale.x, kp_marker.scale.y, kp_marker.scale.z = 0.06, 0.06, 0.06
            for kf in pose.keypoints.values():
                position = kf.x[:3]
                kp_marker.points.append(Point(x=position[0], y=position[1], z=position[2]))
            kp_marker_array.markers.append(kp_marker)

            seg_marker = Marker()
            seg_marker.id = n
            seg_marker.header.frame_id = self.frame_id
            seg_marker.header.stamp = kp_marker.header.stamp
            seg_marker.type = 5  # Line list
            kp_marker.lifetime = Duration(seconds=1).to_msg()
            seg_marker.color = kp_marker.color
            seg_marker.scale.x = 0.05
            for n1, n2 in self.SEG_PAIRS:
                if n1 in pose.keypoints.keys() and n2 in pose.keypoints.keys():
                    p1, p2 = pose.keypoints[n1].x[:3], pose.keypoints[n2].x[:3]
                    seg_marker.points.append(Point(x=p1[0], y=p1[1], z=p1[2]))
                    seg_marker.points.append(Point(x=p2[0], y=p2[1], z=p2[2]))
            seg_marker_array.markers.append(seg_marker)

        self.position_estimation_pub.publish(kp_marker_array)
        self.segments_pub.publish(seg_marker_array)


def main():
    rclpy.init()
    keypoint_fusion = KeypointFusion()
    while True:
        try:
            rclpy.spin(keypoint_fusion)
        except KeyboardInterrupt:
            break
    keypoint_fusion.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
