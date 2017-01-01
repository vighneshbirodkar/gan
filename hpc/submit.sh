JOB_NAME=$1
shift
SCRIPT=$@

echo "Job name = $JOB_NAME"
echo "Script = $SCRIPT"

USER_NAME=$(whoami)

mkdir ./logs

#qsub -N $JOB_NAME -v DL_ARGS -l nodes=1:ppn=2:gpus=1:titan,walltime=47:59:00,pmem=8GB -m abe -M \
#     "$USER_NAME@nyu.edu" -o ./logs/$JOB_NAME.out -e ./logs/$JOB_NAME.err $SCRIPT
