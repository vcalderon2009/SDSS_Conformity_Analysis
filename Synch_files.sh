#!/bin/sh
#
# Description: Synchronizes files and figures from Bender
#
# Parameters
# ----------
# type_file: string
#     Options:
#         - figures
#         - Catalogues

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
    echo "rsync -chavzP --stats "${DIR_B}/data/interim/SDSS/data/mr/Mr19/conformity_output/catl_pickle_files" "${DIR}/data/interim/SDSS/data/mr/Mr19/conformity_output/""
    rsync -chavzP --stats "${DIR_B}/data/interim/SDSS/data/mr/Mr19/conformity_output/catl_pickle_files" "${DIR}/data/interim/SDSS/data/mr/Mr19/conformity_output/"
    echo "rsync -chavzP --stats "${DIR_B}/data/interim/SDSS/mocks/mr/Mr19/conformity_output/catl_pickle_files" "${DIR}/data/interim/SDSS/mocks/mr/Mr19/conformity_output/""
    rsync -chavzP --stats "${DIR_B}/data/interim/SDSS/mocks/mr/Mr19/conformity_output/catl_pickle_files" "${DIR}/data/interim/SDSS/mocks/mr/Mr19/conformity_output/"
fi


