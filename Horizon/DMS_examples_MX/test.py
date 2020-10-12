import pinocchio as pin
import numpy as np

model = pin.buildSampleModelHumanoid()
data = model.createData()

q = pin.randomConfiguration(model,-np.ones(model.nq),np.ones(model.nq))
v = np.random.rand(model.nv)

pin.forwardKinematics(model,data,q,v)
pin.computeJointJacobians(model,data,q)

frame_name = "larm_effector_body"
frame_id = model.getFrameId(frame_name)
frame = model.frames[frame_id]
frame_placement = frame.placement
parent_joint = frame.parent

pin.updateFramePlacements(model,data)
frame_J = pin.getFrameJacobian(model,data,frame_id,pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)

local_to_world_transform = pin.SE3.Identity()
local_to_world_transform.rotation = data.oMf[frame_id].rotation
frame_v = local_to_world_transform.act(frame_placement.actInv(data.v[parent_joint]))

J_dot_v = pin.Motion(frame_J.dot(v))

print("Frame velocity:\n",frame_v)
print("J_dot_v:\n",J_dot_v)