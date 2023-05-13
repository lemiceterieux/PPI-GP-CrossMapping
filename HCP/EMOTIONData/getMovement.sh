for d in */
do 
if [ ! $d == "__pycache__/" ]
then 
aws s3 cp "s3://hcp-openaccess/HCP_900/${d}MNINonLinear/Results/tfMRI_EMOTION_LR/Movement_Regressors.txt" $d/EMOTION_LR/ &
aws s3 cp "s3://hcp-openaccess/HCP_900/${d}MNINonLinear/Results/tfMRI_EMOTION_RL/Movement_Regressors.txt" $d/EMOTION_RL/ &
fi
done
