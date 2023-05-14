execute_process(COMMAND "/home/tumu/anaconda3/envs/stableBaselines/panda-gym/panda_gym/envs/catkin_ws/build/trac_ik_python/catkin_generated/python_distutils_install.sh" RESULT_VARIABLE res)

if(NOT res EQUAL 0)
  message(FATAL_ERROR "execute_process(/home/tumu/anaconda3/envs/stableBaselines/panda-gym/panda_gym/envs/catkin_ws/build/trac_ik_python/catkin_generated/python_distutils_install.sh) returned error code ")
endif()
