
for i in {20380897..20391917}
do
    preserve -c $i
done

squeue --format="%.18i" --me -h | grep -w 20374516.* | xargs scancel