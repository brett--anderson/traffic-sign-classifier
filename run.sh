#!/bin/bash
set -e
. activate carnd-term1

# if [ -z "$1" ]
#   then
#     jupyter notebook --allow-root
# elif [ "$1" == *".ipynb"* ]
#   then
#     jupyter notebook "$1" --allow-root
# else
#     exec "$@"
# fi

jupyter notebook /opt/traffic_sign_predictor/Traffic_Sign_Classifier.ipynb --allow-root