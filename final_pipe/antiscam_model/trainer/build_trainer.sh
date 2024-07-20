BUCKET_URI="gs://mlops-planar-pagoda-425919-f0-unique/"

tar cvf trainer.tar .
gzip trainer.tar
chmod +x trainer.tar.gz
gsutil cp trainer.tar.gz $BUCKET_URI