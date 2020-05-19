from casadi import *
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
import tf as ros_tf
import geometry_msgs.msg
import rospy
from utils.normalize_quaternion import *

class replay_trajectory:
    def __init__(self, dt, joint_list, q_replay, contact_dict={}):
        """
        Contructor
        Args:
            dt:
            joint_list:
            q_replay:
            contact_dict: dictionary containing:
                'frame': force
                TODO: BUG! force expressed in world should be rotated from frame to world! We need to add a setter to provide the possibility to specify forces expressed in world or local

        """
        self.dt = dt
        self.joint_list = joint_list
        self.q_replay = q_replay
        self.contact_dict = contact_dict
        self.force_pub = []


    def publishContactForces(self, time, k):
        i = 0
        for frame in self.contact_dict:
            f_msg = geometry_msgs.msg.WrenchStamped()
            f_msg.header.stamp = time
            f_msg.header.frame_id = frame

            f = self.contact_dict[frame]

            f_msg.wrench.force.x = f[k][0]
            f_msg.wrench.force.y = f[k][1]
            f_msg.wrench.force.z = f[k][2]

            f_msg.wrench.torque.x = 0.
            f_msg.wrench.torque.y = 0.
            f_msg.wrench.torque.z = 0.

            self.force_pub[i].publish(f_msg)
            i += 1


    def replay(self):
        pub = rospy.Publisher('joint_states', JointState, queue_size=10)
        rospy.init_node('joint_state_publisher')
        rate = rospy.Rate(1. / self.dt)
        joint_state_pub = JointState()
        joint_state_pub.header = Header()
        joint_state_pub.name = self.joint_list

        br = ros_tf.TransformBroadcaster()
        m = geometry_msgs.msg.TransformStamped()
        m.header.frame_id = 'world_odom'
        m.child_frame_id = 'base_link'

        nq = np.shape(self.q_replay)[1]
        n_res = np.shape(self.q_replay)[0]

        q_replay = normalize_quaternion(self.q_replay)

        for key in self.contact_dict:
            self.force_pub.append(rospy.Publisher(key+'_forces',geometry_msgs.msg.WrenchStamped, queue_size=1))

        while not rospy.is_shutdown():
            for k in range(int(round(n_res))):
                qk = q_replay[k]

                m.transform.translation.x = qk[0]
                m.transform.translation.y = qk[1]
                m.transform.translation.z = qk[2]
                m.transform.rotation.x = qk[3]
                m.transform.rotation.y = qk[4]
                m.transform.rotation.z = qk[5]
                m.transform.rotation.w = qk[6]

                br.sendTransform((m.transform.translation.x, m.transform.translation.y, m.transform.translation.z),
                                 (m.transform.rotation.x, m.transform.rotation.y, m.transform.rotation.z,
                                  m.transform.rotation.w),
                                 rospy.Time.now(), m.child_frame_id, m.header.frame_id)

                t = rospy.Time.now()
                joint_state_pub.header.stamp = t
                joint_state_pub.position = qk[7:nq]
                joint_state_pub.velocity = []
                joint_state_pub.effort = []
                pub.publish(joint_state_pub)
                self.publishContactForces(t, k)
                rate.sleep()