for INPFILE in *.py; do

	BASENAME=$(basename $INPFILE .gjf)
	INPFILE=$BASENAME.py
	echo "Submitting job for $BASENAME"
	sed -i '10cpython3 '$BASENAME''  A.SUB
	sed -i '130cnp.savetxt("E_'$BASENAME'.txt", deta_E, delimiter=" ")'  $BASENAME
	sed -i '131cnp.savetxt("B_'$BASENAME'.txt", B, delimiter=" ")'  $BASENAME
	RUN_COMMAND="sbatch -J $BASENAME A.SUB"
	eval $RUN_COMMAND

  SLEEPTIME=$(echo "scale=2; $RANDOM/327617" | bc -l)
  echo "Sleeping for $SLEEPTIME seconds to avoid concurrent jobs"
  sleep $SLEEPTIME


done
squeue -u fanxie	
