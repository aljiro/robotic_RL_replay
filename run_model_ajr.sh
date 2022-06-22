#!/bin/sh
python3 robo_RL_WITH_replay_model_FULL.py 
#python3 live_plotting.py &
~/mdk/sim/launch_open_arena.sh 
python3 generate_reward.py 
pause 2
#python miro_controller.py