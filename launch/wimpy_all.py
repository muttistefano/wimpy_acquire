#!/usr/bin/env python3

import os
from os import environ
from os import pathsep
import sys

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution, TextSubstitution
from launch.actions import OpaqueFunction
from ament_index_python import get_package_share_directory
from launch.actions import IncludeLaunchDescription
from launch.actions import GroupAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from scripts import GazeboRosPaths


def generate_launch_description():


    model, plugin, media = GazeboRosPaths.get_paths()

    if 'GAZEBO_MODEL_PATH' in environ:
        model += pathsep+environ['GAZEBO_MODEL_PATH']
    if 'GAZEBO_PLUGIN_PATH' in environ:
        plugin += pathsep+environ['GAZEBO_PLUGIN_PATH']
    if 'GAZEBO_RESOURCE_PATH' in environ:
        media += pathsep+environ['GAZEBO_RESOURCE_PATH']

    env = {'GAZEBO_MODEL_PATH': model,
           'GAZEBO_PLUGIN_PATH': plugin,
           'GAZEBO_RESOURCE_PATH': media}

    print(env)
    # sys.exit()

    declared_arguments = []

    
    declared_arguments.append(
        DeclareLaunchArgument(
            "description_package",
            default_value="wimpy_acquire",
            description="mobile manip description",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "description_file",
            default_value="system.urdf.xacro",
            description="URDF/XACRO description file with the robot.",
        )
    )

    description_package      = LaunchConfiguration("description_package")
    description_file         = LaunchConfiguration("description_file")

    robot_description_content_1 = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution([FindPackageShare(description_package), "urdf", description_file]),
            " ","prefix:=","",
        ]
    )


    robot_description_1  = {"robot_description": robot_description_content_1}

    
    robot_state_publisher_node_1 = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        namespace="",
        output="screen",
        parameters=[robot_description_1],
    )


    spawn_sweepee_1 = Node(package='gazebo_ros', executable='spawn_entity.py',
                            arguments=['-entity', 'sw1','-topic', 'robot_description', '-Y 1.57'],
                            output='screen')
    

    gazebo_server = GroupAction(
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(
                        get_package_share_directory('gazebo_ros'),
                        'launch/gazebo.launch.py')),
                        launch_arguments={'world': os.path.join(get_package_share_directory('wimpy_acquire'),'worlds/world.world'),
                                          'verbose' : 'true' ,
                                          'pause' : 'false'}.items(),
            ),
        ]
    )

    rviz_config_file = PathJoinSubstitution(
        [FindPackageShare("wimpy_acquire"), "rviz", "config.rviz"]
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz_config_file],
    )

    tf_sw1 = Node(
            package="tf2_ros",
            name="tf_sw1",
            executable="static_transform_publisher",
            output="screen" ,
            arguments=["0", "0", "0", "0", "0", "0", "map", "odom"]
        )



########## LAUNCHING


    nodes_to_start = [
        gazebo_server,
        rviz_node,
        TimerAction(
            period=2.0,
            actions=[tf_sw1],
        ),
        TimerAction(
            period=4.0,
            actions=[robot_state_publisher_node_1,spawn_sweepee_1],
        ),
    ]

    return LaunchDescription(declared_arguments + nodes_to_start)


