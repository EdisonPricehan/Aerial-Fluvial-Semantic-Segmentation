# The idea of this script is to run all the steps of the project with a single command.
#
# 1. Train the model
# 2. Inference on test set
# 3. Inference on video
#  3.1 Show results
#
# The reason we're doing this is to make it easier to reproduce some results and to run the whole pipeline.
#

./01.run_train.sh
./02.run_inference.sh
./03.run_video.sh
