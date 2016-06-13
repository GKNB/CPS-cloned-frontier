for id in {0..1000}
do
    perl write_vml.pl $id
    mpirun -np 8 ../NOARCH.x 1 1 -qmp-geom 2 2 2 1 | grep res | perl ./list_res.pl > res.0 > res.$id
done


