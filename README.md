# Horizon
A tiny set of python functions to simplify optimal control in CASADI

Run example centauro_jump
-------------------------
This example to run needs the [casadi_kin_dyn package](https://github.com/ADVRHumanoids/casadi_kin_dyn).

- Goes in the folder ```Horizon/launch/``` and run on a terminal:

```roslaunch centauro_jump.launch```

this will put the template model for centauro (consisting in a box and 4 balls representing the contact feet) and will open rviz.

- Goes in the folder ```Horizon/Horizon/DMS_examples_SX/``` and run:

```python centauro_jump_dt.py```

