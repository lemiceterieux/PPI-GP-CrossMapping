mri_surfcluster --hemi lh --in lh.tval3 --thmin 4.0 --subject fsaverage5 --sum lhtval.txt --ocp lhthval.mgh --surf pial  --annot aparc.a2009s --cwsig lhcwsig.mgh --fwhm 5
mri_surfcluster --hemi rh --in rh.tval3 --thmin 4.0 --subject fsaverage5 --sum rhtval.txt --ocp rhthval.mgh --surf pial  --annot aparc.a2009s --cwsig rhcwsig.mgh --fwhm 5
