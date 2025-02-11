echo "Now start experiment"
start=$(date "+%s")

seq 1 1 1 | parallel python LDPC_ex2.py 4 3 0.1 0.2 3 18000 1
python plot_result_LDPC_ex2.py 1

now=$(date "+%s")
time=$((now-start))

echo "The expreiment has done!"
echo "time used: $time seconds"
