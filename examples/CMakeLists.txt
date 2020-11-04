add_custom_target(generate_py_from_nb ALL
                jupyter nbconvert --to python ${CMAKE_CURRENT_SOURCE_DIR}/notebook/inv_pendulum.ipynb --output-dir='${CMAKE_CURRENT_BINARY_DIR}'
                )

install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/inv_pendulum.py
        DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}) 
