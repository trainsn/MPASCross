#PBS -N MPASCross 
#PBS -l walltime=1:00:00
#PBS -l nodes=1:ppn=1
#PBS -j oe

make

./MPASCross BwsA0.00_.nc
./MPASCross BwsA1.00_.nc
./MPASCross BwsA3.00_.nc
./MPASCross BwsA5.00_.nc