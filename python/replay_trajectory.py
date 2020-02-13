from casadi import *
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
import tf as ros_tf
import geometry_msgs.msg
import rospy

class replay_trajectory:
    def __init__(self, ns, dt, joint_list, q_replay):
        self.ns = ns
        self.dt = dt
        self.joint_list = joint_list
        self.q_replay = q_replay

    def normalize(self, v):
        return v / np.linalg.norm(v)

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

        while not rospy.is_shutdown():
            for k in range(self.ns):
                qk = self.q_replay[k]

                m.transform.translation.x = qk[0]
                m.transform.translation.y = qk[1]
                m.transform.translation.z = qk[2]
                quat = [qk[3], qk[4], qk[5], qk[6]]
                quat = self.normalize(quat)
                m.transform.rotation.x = quat[0]
                m.transform.rotation.y = quat[1]
                m.transform.rotation.z = quat[2]
                m.transform.rotation.w = quat[3]

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







