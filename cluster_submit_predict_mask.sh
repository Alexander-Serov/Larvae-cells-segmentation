#! /bin/bash


## Constants
# max_D_case_number=4
# max_f_case_number=5

## Initialize
str_date=`date +"%F_---_%H-%M-%S "`

echo "Submitting tasks to kernels..."

# for D_case_number in $(seq 1 $max_D_case_number)
# do	
	# for f_case_number in $(seq 1 $max_f_case_number)
	# do
		# echo "Submitting task with D=${D_case_number}, f=${f_case_number} to kernels..."
		# nohup python2.7 simulate_one_trajectory.py $D_case_number $f_case_number > "log_D=${D_case_number}_f=${f_case_number}.log" &
		module load Python/2.7.11
		module load cuda/8.0.0 cudnn/v5
		module load tensorflow/1.0.0-py2
		# module load test/tensorflow/0.11.0rc2
		srun -o "placenta_predict_${str_date}.out" -e "placenta_predict_${str_date}.err" -J "plac_pr5" --qos=gpu --mem=21GB python predict_mask.py  &
		# --qos=gpu --gres=gpu:teslaM40:1 --mem=21GB python predict_mask.py  &
		# echo "Task submitted"
	# done
# done

echo "All tasks submitted successfully!"



