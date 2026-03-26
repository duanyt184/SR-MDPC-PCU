start=$(date +%s)

rm -rf ./evaluation_code/result
mkdir ./evaluation_code/result
CUDA_VISIBLE_DEVICES=0 python3 test.py 2>&1 | tee test.log
cd ./evaluation_code
bash eval_mydata.sh
python show_p2f.py 2>&1 | tee -a ../test.log

end=$(date +%s)
time_consumed=$(( end - start ))
echo "Time elapsed: $time_consumed seconds"
