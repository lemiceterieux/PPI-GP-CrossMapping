for f in `aws s3 ls "s3://hcp-openaccess/HCP_900/"|grep PRE|awk '{print $2}'`
do
if [ ! -d "$f" ]; then
if ((`aws s3 ls  "s3://hcp-openaccess/HCP_900/${f}MNINonLinear/Results/tfMRI_EMOTION_LR/"|wc -l` >= 3)); then
mkdir $f
cd $f
mkdir EMOTION_LR
cd EMOTION_LR
aws s3 cp "s3://hcp-openaccess/HCP_900/${f}MNINonLinear/Results/tfMRI_EMOTION_LR/tfMRI_EMOTION_LR.nii.gz" .
aws s3 sync "s3://hcp-openaccess/HCP_900/${f}MNINonLinear/Results/tfMRI_EMOTION_LR/EVs" ./EVs
echo "$f LR" 
cd ../
mkdir EMOTION_RL
cd EMOTION_RL
aws s3 cp "s3://hcp-openaccess/HCP_900/${f}MNINonLinear/Results/tfMRI_EMOTION_RL/tfMRI_EMOTION_RL.nii.gz" .
aws s3 sync "s3://hcp-openaccess/HCP_900/${f}MNINonLinear/Results/tfMRI_EMOTION_RL/EVs" ./EVs
echo "$f RL"
cd ../
cd ../
fi
fi
done
