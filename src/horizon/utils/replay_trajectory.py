from casadi import *
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
import tf as ros_tf
import geometry_msgs.msg
import rospy
from utils.normalize_quaternion import *

class replay_trajectory:
    def __init__(self, dt, joint_list, q_replay):
        self.dt = dt
        self.joint_list = joint_list
        self.q_replay = q_replay

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

                joint_state_pub.header.stamp = rospy.Time.now()
                joint_state_pub.position = qk[7:nq]
                joint_state_pub.velocity = []
                joint_state_pub.effort = []
                pub.publish(joint_state_pub)
                rate.sleep()