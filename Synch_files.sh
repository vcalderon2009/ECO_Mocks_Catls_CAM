#!/bin/sh
#
# Description: Synchronizes files and figures from Bender
#
# Parameters
# ----------
# file_opt: string
#     Options:
#         - catalogues

# Defining Directory
DIR_LOCAL="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ZOIDBERG_SSH="caldervf@zoidberg.phy.vanderbilt.edu:"
DIR_ZOIDBERG="${ZOIDBERG_SSH}/home/www/groups/data_eco_vc/ECO_CAM/Mock_Catalogues"
# option
file_opt=$1
echo "\n==> Option: ${file_opt}"
## Help option
usage="Synch_files.sh [-h] [file_opt] -- Program that synchronizes files between 'local' and Bender
where:
    -h           show this help text

    Options for 'file_opt':
        - 'catalogues'  Synchronizes catalogues files in 'data' folder"

if [[ ${file_opt} == '-h' ]]; then
  echo "==> Usage: $usage\n"
  # exit 0
fi
##
## Synchronizing
# Catalogues
if [[ ${file_opt} == 'catalogues' ]]; then
    echo "==> rsync -chavzP --stats "${DIR_LOCAL}/data/processed/TAR_files/"  "${DIR_ZOIDBERG}"\n"
    rsync -chavzP --stats "${DIR_LOCAL}/data/processed/TAR_files/"  "${DIR_ZOIDBERG}"
fi