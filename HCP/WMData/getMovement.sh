for d in */
do 
if [ ! $d == "__pycache__/" ]
then 
aws s3 cp "s3://hcp-openaccess/HCP_900/${d}MNINonLinear/Results/tfMRI_WM_LR/Movement_Regressors.txt" $d/WM_LR/ &
aws s3 cp "s3://hcp-openaccess/HCP_900/${d}MNINonLinear/Results/tfMRI_WM_RL/Movement_Regressors.txt" $d/WM_RL/ &
fi
done
