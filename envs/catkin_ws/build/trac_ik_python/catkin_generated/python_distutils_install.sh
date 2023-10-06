#!/bin/sh

if [ -n "$DESTDIR" ] ; then
    case $DESTDIR in
        /*) # ok
            ;;
        *)
            /bin/echo "DESTDIR argument must be absolute... "
            /bin/echo "otherwise python's distutils will bork things."
            exit 1
    esac
fi

echo_and_run() { echo "+ $@" ; "$@" ; }

echo_and_run cd "/home/tumu/anaconda3/envs/stableBaselines/panda-gym/panda_gym/envs/catkin_ws/src/trac_ik/trac_ik_python"

# ensure that Python install destination exists
echo_and_run mkdir -p "$DESTDIR/home/tumu/anaconda3/envs/stableBaselines/panda-gym/panda_gym/envs/catkin_ws/install/lib/python3/dist-packages"

# Note that PYTHONPATH is pulled from the environment to support installing
# into one location when some dependencies were installed in another
# location, #123.
echo_and_run /usr/bin/env \
    PYTHONPATH="/home/tumu/anaconda3/envs/stableBaselines/panda-gym/panda_gym/envs/catkin_ws/install/lib/python3/dist-packages:/home/tumu/anaconda3/envs/stableBaselines/panda-gym/panda_gym/envs/catkin_ws/build/trac_ik_python/lib/python3/dist-packages:$PYTHONPATH" \
    CATKIN_BINARY_DIR="/home/tumu/anaconda3/envs/stableBaselines/panda-gym/panda_gym/envs/catkin_ws/build/trac_ik_python" \
    "/home/tumu/anaconda3/envs/stableBaselines/bin/python" \
    "/home/tumu/anaconda3/envs/stableBaselines/panda-gym/panda_gym/envs/catkin_ws/src/trac_ik/trac_ik_python/setup.py" \
     \
    build --build-base "/home/tumu/anaconda3/envs/stableBaselines/panda-gym/panda_gym/envs/catkin_ws/build/trac_ik_python" \
    install \
    --root="${DESTDIR-/}" \
    --install-layout=deb --prefix="/home/tumu/anaconda3/envs/stableBaselines/panda-gym/panda_gym/envs/catkin_ws/install" --install-scripts="/home/tumu/anaconda3/envs/stableBaselines/panda-gym/panda_gym/envs/catkin_ws/install/bin"
