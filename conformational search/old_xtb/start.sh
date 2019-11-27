for INPFILE in *.gjf
do
		 
         BASENAME=$(basename $INPFILE)
         INPFILE=$BASENAME
         echo $BASENAME
         mkdir $BASENAME.F
		 a=$(wc -l < $BASENAME)
		 noa=$(expr $a - 8 )
		 tend=$(expr $noa + 11 )
	 cp xtb_exe.tgz test.xyz A.SUB $BASENAME $BASENAME.F/
	 cd $BASENAME.F/
	 mkdir 4000 300
	 cp xtb_exe.tgz test.xyz A.SUB $BASENAME 4000/
	 cp xtb_exe.tgz test.xyz A.SUB $BASENAME 300/
	 cd 4000
	 tar -xzvf xtb_exe.tgz
	 cat $BASENAME | tail -n +8 | head -n $noa > a.txt #start line and end line
	 sed -i '1i\'$noa a.txt  #number of atoms
	 sed -i '1i\'$noa a.txt  #number of atoms
	 cat a.txt test.xyz > input.xyz
	 sed -i $tend'c\tend         4000       # highest siman annealing temperature (very system specific)' input.xyz # noa + 11 
	 sbatch A.SUB
	 cd ../300
	 tar -xzvf xtb_exe.tgz
	 cat $BASENAME | tail -n +8 | head -n $noa > a.txt #start line and end line
	 sed -i '1i\'$noa a.txt   #number of atoms
	 sed -i '1i\'$noa a.txt   #number of atoms
	 cat a.txt test.xyz > input.xyz
	 sed -i $tend'c\tend         300       # highest siman annealing temperature (very system specific)' input.xyz #noa + 11
	 sbatch A.SUB
	 cd ../../
done

sq
