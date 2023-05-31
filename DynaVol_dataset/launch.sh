for ((i=0;i<100;i++))
do
    docker run --rm --interactive   --user $(id -u):$(id -g)      --volume "$(pwd):/kubric"     kubricdockerhub/kubruntu /usr/bin/python3 DynaVol_syn_shape.py  --job_dir "./built_datasets/DynaVol/cpu-$i" 
done
