pip3 install tensorflow&
pip3 install keras
wait

tpython="LD_LIBRARY_PATH="/home/zlstg1/cding0622/my_libc_env/libc6_2.17/lib/x86_64-linux-gnu/:/home/zlstg1/cding0622/my_libc_env/usr/lib64/:/home/zlstg1/cding0622/gccLocal/usr/local/lib64"  /home/zlstg1/cding0622/my_libc_env/libc6_2.17/lib/x86_64-linux-gnu/ld-2.17.so `which /home/zlstg1/cding0622/python3.5/opt/python3.5/bin/python3`"
wait
shopt -s expand_aliases
source ~/.bash_profile
wait

tpython trainer.py
