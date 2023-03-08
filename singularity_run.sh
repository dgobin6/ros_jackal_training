#!/bin/bash
export ROS_HOSTNAME=localhost
export ROS_MASTER_URI=http://localhost:11311
export BUFFER_PATH=./local_buffer

singularity exec -i --nv -n --network=none -p -B `pwd`:/jackal_ws/src/ros_jackal_training -B ${BUFFER_PATH}:/local_buffer ${1} /bin/bash /jackal_ws/src/ros_jackal_training/entrypoint.sh ${@:2}
