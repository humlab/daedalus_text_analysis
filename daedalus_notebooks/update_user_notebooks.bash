#!/bin/bash

function copy_notebook {
	target_user=$1
	notebook=$2
	cp -f ${notebook} /home/${target_user}/notebooks/${notebook}
	chown ${target_user}.developer /home/${target_user}/notebooks/${notebook}
}

#copy_notebook fredrik_noren explore-vector-space.ipynb
#copy_notebook fredrik_noren explore-topic-models.ipynb

#copy_notebook pelle_snickars explore-vector-space.ipynb
#copy_notebook pelle_snickars explore-topic-models.ipynb

copy_notebook johan_jarlbrink explore-vector-space.ipynb
copy_notebook johan_jarlbrink explore-topic-models.ipynb

