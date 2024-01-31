import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():

    camera_launch_path = (
        os.path.join(get_package_share_directory('realsense2_camera'), 'launch', 'rs_multi_camera_launch.py'))
    cameras = IncludeLaunchDescription(
        launch_description_source=PythonLaunchDescriptionSource(
            launch_file_path=camera_launch_path
        ),
        launch_arguments={
            'rgb_camera.profile1': '640x480x30',
            'rgb_camera.profile2': '640x480x30',
            'depth_module.profile1': '640x480x30',
            'depth_module.profile2': '640x480x30',
            'pointcloud.enable1': 'True',
            'pointcloud.enable2': 'True',
            'align_depth.enable1': 'True',
            'align_depth.enable2': 'True',
            'color_qos1': 'SENSOR_DATA',
            'color_qos2': 'SENSOR_DATA',
            'depth_qos1': 'SENSOR_DATA',
            'depth_qos2': 'SENSOR_DATA',
            'serial_no1': '_151422250087',
            'serial_no2': '_927522073083'
        }.items()
    )

    base_link_cam1_tf = Node(package="tf2_ros",
                             executable="static_transform_publisher",
                             arguments=["0", "0", "2", "0", "0", "0", "1",
                                        "base_link", "camera1_link"]
                             )

    cam1_cam2_tf = Node(package="tf2_ros",
                        executable="static_transform_publisher",
                        # arguments=["0.9898", "-0.8532", "0.2291", "-0.2486", "-0.0589", "0.7552", "0.6037",\
                        arguments=['0.962300', '-0.885500', '0.225300', 
                                   '-0.247400', '-0.059500', '0.734200', '0.629500',
                                   "camera1_link", "camera2_link"]
                        )

    yolov8_3d_path = os.path.join(get_package_share_directory('yolov8_bringup'), 'launch', 'yolov8_3d.launch.py')
    pose_detection_1 = IncludeLaunchDescription(
        launch_description_source=PythonLaunchDescriptionSource(
            launch_file_path=yolov8_3d_path,
        ),
        launch_arguments={
            'namespace': 'yolo1',
            'input_image_topic': '/camera1/color/image_raw',
            'input_depth_topic': '/camera1/aligned_depth_to_color/image_raw',
            'input_depth_info_topic': '/camera1/aligned_depth_to_color/camera_info',
            'model': 'yolov8m-pose'
        }.items()
    )
    pose_detection_2 = IncludeLaunchDescription(
        launch_description_source=PythonLaunchDescriptionSource(
            launch_file_path=yolov8_3d_path,
        ),
        launch_arguments={
            'namespace': 'yolo2',
            'input_image_topic': '/camera2/color/image_raw',
            'input_depth_topic': '/camera2/aligned_depth_to_color/image_raw',
            'input_depth_info_topic': '/camera2/aligned_depth_to_color/camera_info',
            'model': 'yolov8m-pose'
        }.items()
    )

    return LaunchDescription([
        cameras,
        base_link_cam1_tf,
        cam1_cam2_tf,
        pose_detection_1,
        pose_detection_2
    ])
