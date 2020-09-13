#!/bin/bash

if [ "$#" -ne 0 ]; then
    echo "usage: bash run.sh"
    exit 0
fi

# clear log
#echo "" > log.out

#exec 3>&1 4>&2
#trap 'exec 2>&4 1>&3' 0 1 2 3
#exec 1>log.out 2>&1
# Everything below will go to the file 'log.out':
set -e
set -x

script_path=$PWD
locations_path=$PWD/locations
#save_path=/media/benjamin/Karla/Carla/dataset
save_path=/home/benjamin/dataset

mkdir -p "$save_path"

killall -9 CarlaUE4-Linux-Shipping || true
sleep 3

maps=( Town01 Town02 Town03 Town04 Town05 Town06 Town07 Town10HD )

for map in ${maps[*]}; do
    start=`date +%s`
    echo -e "\n\e[31mMap: $map\e[0m"

    echo -e "\e[32mLaunching server\e[0m"
    (DISPLAY= "/opt/carla-simulator/CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping" CarlaUE4 -opengl)&
    carla_pid=$(pidof CarlaUE4-Linux-Shipping)
    sleep 30

    echo -e "\n\e[32mChanging Map\e[0m"
    cd "$script_path"
    mkdir -p "$save_path"/"$map"
    python3 "$script_path"/config.py -m "$map"
    sleep 4
    if [ "$map" = "Town10HD" ]; then
      sleep 30
    fi


    echo -e "\n\e[32mStarting Script\e[0m"
    cd "$script_path"
    python3 "$script_path"/create_data.py -d "$save_path"/"$map" --yaml_path "$locations_path" --map "$map" &&
    sleep 2

    # check if create_data wrote 1 or 0 to this file (representing the exit status)
    if grep -q 1 "$locations_path/exit_code"; then
      echo -e "\n\e[32mcreate_data failed!\e[0m"
      exit 1
    fi

    # remove the exit code file, it will be recreated in the next run
    rm $locations_path/exit_code

    echo -e "\n\e[32mKilling server\e[0m"
    kill -9 "$carla_pid"
    sleep 2
    end=`date +%s` && runtime=$((end-start))
    echo -e "\nTime for $map = $((runtime / 60)) Min $((runtime % 60)) Sek."

    #cd "$save_path"
    #tar -zcvf "$map".tar.gz "$Save_path"/"$map" &&  rm -rf "$Save_path"/"$map"/*

done

echo -e "\n\e[95mSuccessfully created dataset.\e[0m"
