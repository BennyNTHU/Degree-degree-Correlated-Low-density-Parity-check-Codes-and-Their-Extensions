echo "Now start experiment"
start=$(date "+%s")

for q in $(seq 0.0 .1 1.0); do
    for i in {1..100} 
    do
        python negative_correlation.py $q
        echo "experiment: q=$q #$i is done!"
    done
done

python plot_result.py

now=$(date "+%s")
time=$((now-start))

echo "The expreiment has done!"
echo "time used: $time seconds"