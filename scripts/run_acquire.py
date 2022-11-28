#!/usr/bin/env python3

from geometry_msgs.msg import TransformStamped, Twist
from sensor_msgs.msg import LaserScan
import rclpy
import time
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from tf2_ros import TransformBroadcaster,TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from turtlesim.msg import Pose
from random import uniform
import threading
import numpy as np
from tf_transformations import euler_from_quaternion, quaternion_from_euler
from gazebo_msgs.srv import SetEntityState
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
import os,sys
import threading

class Acquire_class(Node):

    def __init__(self):
        super().__init__('run_acquire')

        self.declare_parameter('turtlename', 'turtle')
        self.turtlename = self.get_parameter(
            'turtlename').get_parameter_value().string_value

        self.br          = TransformBroadcaster(self)
        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # self.loop_rate_  = self.create_rate(10, self.get_clock())

        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.SYSTEM_DEFAULT,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )

        self.service_cb   = MutuallyExclusiveCallbackGroup()
        self.time_cb      = MutuallyExclusiveCallbackGroup()
        # self.subscription = self.create_subscription(LaserScan,'/front_laser_plugin/out',self.listener_callback,qos_profile)
        self.publisher_   = self.create_publisher(Twist, 'custom_cmd_vel', 1)
        # self.timer        = self.create_timer(1, self.move_robot,callback_group=self.time_cb)
        self.shuffle_srv  = self.create_client(SetEntityState, '/set_entity_state',callback_group=self.service_cb)
        
        self.laser_log_   = [None] * 10
        self.cnt_         = 0
        self.save_cnt_    = 0
        self.fld_cnt      = 0
        self.mutex        = threading.Lock()

        fld_cnt_trial   = len(os.listdir("/home/blanker/wimpy/src/wimpy_acquire/logs/"))
        self.trial_root = "/home/blanker/wimpy/src/wimpy_acquire/logs/trial" + str(fld_cnt_trial)
        os. mkdir(self.trial_root)

        self.curr_tf_     = [0,0,0]
        self.pos_         = [0,0,0]
        self.run_th = threading.Thread(target=self.run)
        self.run_th.start()
        # self.run()
        
    def listener_callback(self, msg):
        try:
            self.mutex.acquire(blocking=True)
            self.laser_log_[self.cnt_%10] = msg.ranges
            self.cnt_ = self.cnt_ + 1
        except Exception as e:
            print(e)
        finally:
            self.mutex.release()
        
    def log_data(self):
        self.mutex.acquire(blocking=True)
        np.save(self.act_folder + "/ls_" + str(self.save_cnt_),np.asarray(self.laser_log_))
        np.save(self.act_folder + "/tf_" + str(self.save_cnt_),np.asarray(self.curr_tf_))
        self.save_cnt_ = self.save_cnt_ + 1
        self.mutex.release()

    def __del__(self):
        self.stop_move()

    def stop_move(self):
        msg = Twist()
        msg.linear.x  = 0.0
        msg.linear.y  = 0.0
        msg.angular.z = 0.0
        self.publisher_.publish(msg)

    def move_robot(self):
        print("moving")
        msg = Twist()
        new_pos = [uniform(-0.05, 0.05),uniform(-0.05, 0.05),uniform(-0.05, 0.05)]

        x_vel = (self.pos_[0] - new_pos[0])/0.2
        y_vel = (self.pos_[1] - new_pos[1])/0.2
        z_vel = (self.pos_[2] - new_pos[2])/0.2

        msg.linear.x  = x_vel
        msg.linear.y  = y_vel

        msg.angular.z = z_vel

        now = time.time()
        while (time.time()-now)<0.02:
            self.publisher_.publish(msg)
            time.sleep(0.001)
        self.stop_move()

        try:
            trans = self.tf_buffer.lookup_transform(
                "front_laser",
                "map",
                rclpy.time.Time())
        except TransformException as ex:
            self.get_logger().info(
                f'Could not transform ')
            return
        
        ang_w = euler_from_quaternion([trans.transform.rotation.x,trans.transform.rotation.y,trans.transform.rotation.z,trans.transform.rotation.w])[2]
        self.curr_tf_ = [trans.transform.translation.x,trans.transform.translation.y,ang_w]
        print("moved")
        self.log_data()
        self.pos_ = new_pos
        time.sleep(0.1)

    def shuffle(self):
        # self.fld_cnt = len(os.listdir("/home/blanker/wimpy/src/wimpy_acquire/logs/"))
        self.act_folder = str(self.trial_root) + "/" + str(self.fld_cnt)
        os. mkdir(self.act_folder)
        self.fld_cnt = self.fld_cnt + 1
        self.save_cnt_ = 0
        print("shuffle")

        req = SetEntityState.Request()
        for el in range(10):
            req.state.name = "b" + str(el)  
            req.state.pose.position.x = uniform(-6, 6)
            req.state.pose.position.y = uniform(1.0, 4)
            q = quaternion_from_euler(0, 0, uniform(-3.14,3.14))
            req.state.pose.orientation.w = q[3]
            req.state.pose.orientation.x = q[0]
            req.state.pose.orientation.y = q[1]
            req.state.pose.orientation.z = q[2]
            resp = self.shuffle_srv.call(req)
            time.sleep(0.02)
        for el in range(10):
            req.state.name = "t" + str(el)  
            req.state.pose.position.x = uniform(-6, 6)
            req.state.pose.position.y = uniform(1.0, 4)
            resp = self.shuffle_srv.call(req)
            time.sleep(0.02)
        print("shuffle ended")
        time.sleep(1)

    def run(self):
        while rclpy.ok():
            self.shuffle()
            for i in range(100):
                self.move_robot()




def main():
    rclpy.init()
    
    node = Acquire_class()

    # executor = MultiThreadedExecutor()
    # executor.add_node(node)
    minimal_publisher = Acquire_class()

    rclpy.spin(minimal_publisher)
    try:
        # rclpy.spin(node)
        minimal_publisher.spin()
    # except KeyboardInterrupt:
    #     node.stop_move()
    finally:
        del node

    rclpy.shutdown()

if __name__ == "__main__":
    main()
