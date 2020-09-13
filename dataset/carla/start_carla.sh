#!/bin/bash

if ! pidof CarlaUE4-Linux-Shipping CarlaUE4-Linux-Shipping > /dev/null;then
  echo "Carla is not Running. Starting!"
  cd /opt/carla-simulator/CarlaUE4/Binaries/Linux &&
  #setsid ./CarlaUE4-Linux-Shipping -quality-level=High >/dev/null 2>&1 < /dev/null &
  DISPLAY=
  setsid "/opt/carla-simulator/CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping" CarlaUE4 -opengl &
  sleep 8
fi

cd /opt/carla-simulator/PythonAPI/examples/