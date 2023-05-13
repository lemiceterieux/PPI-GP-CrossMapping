for f in `aws s3 ls "s3://hcp-openaccess/HCP_900/"|grep PRE|awk '{print $2}'`
do
if [ ! -d "$f" ]; then
if ((`aws s3 ls  "s3://hcp-openaccess/HCP_900/${f}MNINonLinear/Results/tfMRI_WM_LR/"|wc -l` >= 3)); then
mkdir $f
cd $f
mkdir WM_LR
cd WM_LR
aws s3 cp "s3://hcp-openaccess/HCP_900/${f}MNINonLinear/Results/tfMRI_WM_LR/tfMRI_WM_LR.nii.gz" .
aws s3 sync "s3://hcp-openaccess/HCP_900/${f}MNINonLinear/Results/tfMRI_WM_LR/EVs" ./EVs
echo "$f LR" 
cd ../
mkdir WM_RL
cd WM_RL
aws s3 cp "s3://hcp-openaccess/HCP_900/${f}MNINonLinear/Results/tfMRI_WM_RL/tfMRI_WM_RL.nii.gz" .
aws s3 sync "s3://hcp-openaccess/HCP_900/${f}MNINonLinear/Results/tfMRI_WM_RL/EVs" ./EVs
echo "$f RL"
cd ../
cd ../
fi
fi
done
