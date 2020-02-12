cmake_minimum_required(VERSION 2.8.3)
project(Horizon)

find_package(catkin REQUIRED)

catkin_package()

include_directories( )

## Mark other files for installation (e.g. launch and bag files, etc.)
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
  FILES_MATCHING PATTERN "*.launch"
)
                                                      

install(DIRECTORY configs/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/configs
  FILES_MATCHING PATTERN "*.urdf"
)  


install(PROGRAMS
   python/horizon.py
   python/constraints.py
   python/previewer.py
   python/inverse_dynamics.py
   python/multiple_shooting_example.py
   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
