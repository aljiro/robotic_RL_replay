#!/bin/sh
python3 robot_replay_model.py &
#python3 live_plotting.py &
sh ~/mdk/sim/launch_open_arena.sh &
python3 generate_reward.py &
pause 2
#python miro_controller.py