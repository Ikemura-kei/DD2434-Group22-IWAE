#! /bin/bash

declare -a arr=(
               "cfg/larger_bs_omniglot_vae_k1.yml" 
               "cfg/larger_bs_omniglot_vae_k5.yml" 
               "cfg/larger_bs_omniglot_vae_k50.yml" 
               "cfg/larger_bs_mnist_vae_k1.yml" 
               "cfg/larger_bs_mnist_vae_k5.yml" 
               "cfg/larger_bs_mnist_vae_k50.yml" 
                )

for cfg in "${arr[@]}"
do
   echo "$cfg"
   
   python src/main.py --cfg $cfg
done
