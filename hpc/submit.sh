JOB_NAME=$1
shift
export COMMAND="$@ --name $JOB_NAME"


SCRIPT="$SCRIPT --name $JOB_NAME"
echo "Job name = $JOB_NAME"
echo "Command = $COMMAND"

USER_NAME=$(whoami)

mkdir -p ./logs


qsub -v COMMAND -N $JOB_NAME -l nodes=1:ppn=2:gpus=1:titan,walltime=47:59:00,pmem=8GB\
     -m abe -M "$USER_NAME@nyu.edu" -o ./logs/$JOB_NAME.out -e ./logs/$JOB_NAME.err \
     ./script.sh
