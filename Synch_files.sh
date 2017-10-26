#!/bin/sh
#
# Description: Synchronizes files and figures from Bender
#
# Parameters
# ----------
# type_file: string
#     Options:
#         - figures
#         - catalogues

# Defining Directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DIR_B="caldervf@bender.accre.vanderbilt.edu:/fs1/caldervf/Repositories/Large_Scale_Structure/SDSS/SDSS_Conformity_Analysis"
file_opt=$1
echo "Option: ${file_opt}"
# Displaying Input Parameters
# Synchronizing
if [[ ${file_opt} == 'figures' ]]; then
    echo "rsync -chavzP --stats --delete "${DIR_B}/reports/figures" "${DIR}/reports""
    rsync -chavzP --stats --delete "${DIR_B}/reports/figures" "${DIR}/reports"
fi
if [[ ${file_opt} == 'catalogues' ]]; then
    echo "rsync -chavzP --stats "${DIR_B}/data/processed/SDSS" "${DIR}/data/processed/""
    rsync -chavzP --stats "${DIR_B}/data/processed/SDSS" "${DIR}/data/processed/"
fi



