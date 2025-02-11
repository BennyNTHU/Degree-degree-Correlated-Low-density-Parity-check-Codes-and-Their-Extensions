echo "Now start numerical verification"
start=$(date "+%s")

for i in {2,5,10,20,30,40,50,60,70,80,90,100,150,200}
do
    # python theoretical.py 3 2 0.1 $i
    python theoretical.py 3 2 0.5 $i
    # python theoretical.py 3 2 0.9 $i
done

now=$(date "+%s")
time=$((now-start))

echo "The verification has done!"
echo "time used: $time seconds"
