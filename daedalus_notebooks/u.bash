#!/bin/bash

declare -a notebooks=("explore-vector-space.ipynb" "explore-topic-models.ipynb")
usernames=("fredrik_noren" "pelle_snickars" "johan_jarlbrink")

function copy_to_user() {
    target_user=$1
    mkdir -p /home/${target_user}/notebooks
    chown ${target_user}.developer /home/${target_user}/notebooks
    for i in "${notebooks[@]}";
    do
        echo "$i"
	notebook="$i"
        echo cp -f ${notebook} /home/${target_user}/notebooks/${notebook}
        echo chown ${target_user}.developer /home/${target_user}/notebooks/${notebook}
    done
}


copy_to_user "andreas"

#function copy_notebook {
#
	#target_user=$1
	#notebook=$2
##
	#cp -f ${notebook} /home/${target_user}/notebooks/${notebook}
	#chown ${target_user}.developer /home/${target_user}/notebooks/${notebook}
#}

#copy_notebook fredrik_noren explore-vector-space.ipynb
#copy_notebook fredrik_noren explore-topic-models.ipynb

#copy_notebook pelle_snickars explore-vector-space.ipynb
#copy_notebook pelle_snickars explore-topic-models.ipynb

#copy_notebook johan_jarlbrink explore-vector-space.ipynb
#copy_notebook johan_jarlbrink explore-topic-models.ipynb

