for INPFILE in *.gjf
do

         BASENAME=$(basename $INPFILE)
         INPFILE=$BASENAME
         echo $BASENAME
		 a=$(wc -l < $BASENAME)
		 noa=$(expr $a - 8 )
		 splitline=$(expr $noa + 2 )
         cd $BASENAME.F
		 
		 cd 300
		 mkdir gaussian

		 cp xtbsiman.log gaussian/

		 cd gaussian/

		 split -l $splitline xtbsiman.log -d -a 3 #noa +2

		 rm -rf xtbsiman.log
		 
		 for  INPFILE1 in *
                       do
                         BASENAME1=$(basename $INPFILE1)
			 INPFILE1=$BASENAME1
		         echo $BASENAME1
			 sed -i '1d' $BASENAME1
			 sed -i '1d' $BASENAME1

			 cd ../../../

                         cp method.txt $BASENAME.F/300/gaussian/

                         cd $BASENAME.F/300/gaussian	

			 cat method.txt $BASENAME1 > $BASENAME1.gjf
			 
			 for k in $(seq 1 3)
	                      do
		                 echo "" >> $BASENAME1.gjf
	                      done
				   
		      done
		  
		  cd ../../4000
		  mkdir gaussian

		  cp xtbsiman.log gaussian/

		  cd gaussian/

		  split -l $splitline xtbsiman.log -d -a 3 #noa +2

		  rm -rf xtbsiman.log
		 
		  for  INPFILE2 in *
                      do
                         BASENAME2=$(basename $INPFILE2)
			 INPFILE2=$BASENAME2
		         echo $BASENAME2
			 sed -i '1d' $BASENAME2
			 sed -i '1d' $BASENAME2

			 cd ../../../

                         cp method.txt $BASENAME.F/4000/gaussian/

                         cd $BASENAME.F/4000/gaussian			 

			 cat method.txt $BASENAME2 > $BASENAME2.gjf
			 
			 for k in $(seq 1 3)
	                     do
		               echo "" >> $BASENAME2.gjf
	                     done
				   
		      done
		  
		  cd ../../../
		  
		
		 	
done

mkdir summary
k=0
for INPFILE in *.gjf
do
         k=$(($k+1))

         BASENAME=$(basename $INPFILE)
         INPFILE=$BASENAME
         echo $BASENAME
         cd $BASENAME.F
		 
	 cd 300/gaussian/
		 
	 for INPFILE1 in *.gjf
         do

             BASENAME1=$(basename $INPFILE1)
             INPFILE1=$BASENAME1
             echo $BASENAME1
             cp $BASENAME1 $k$BASENAME1
	     mv $k$BASENAME1 ../../../summary
	 done

	 k=$(($k+1))
		 
	 cd ../../4000/gaussian/
		 
	 for INPFILE2 in *.gjf
         do

             BASENAME2=$(basename $INPFILE2)
             INPFILE2=$BASENAME2
             echo $BASENAME2
             cp $BASENAME2 $k$BASENAME2
	     mv $k$BASENAME2 ../../../summary
	 done
		 
	 cd ../../../
		 
		 		 	
done

cp extractfiles.py summary/
cd summary
python3.7 extractfiles.py


