echo "Inside script.sh"
echo "Command = $COMMAND"

module load torch/gnu/20160623
cd /scratch/vnb222/code/gan
eval "$COMMAND"
