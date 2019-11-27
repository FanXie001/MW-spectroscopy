input=$1
noa=$2
a=2
splitline=`expr $noa + $a`

mkdir gjf
cp $input gjf

cd gjf
split -l $splitline $input -d -a 3 #noa +2
rm -rf $input

for  INPFILE1 in *  
do
		echo $INPFILE1
		sed -i '1d' $INPFILE1
		sed -i '1d' $INPFILE1
		cat ../method.txt $INPFILE1 > $INPFILE1.gjf
		
		for k in $(seq 1 1)  
		do
		    echo "" >> $INPFILE1.gjf
	    	done
done
