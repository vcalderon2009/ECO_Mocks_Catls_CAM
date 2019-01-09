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
SSH_USERNAME="caldervf"
SSH_HOST="zoidberg.phy.vanderbilt.edu"
SSH_PATH="/home/www/groups/data_eco_vc/ECO_CAM/Mock_Catalogues"
# Commands to Send over SSH
CATL_COPY_COMMAND="${SSH_USERNAME}@${SSH_HOST}:${SSH_PATH}"
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
    # Removes previous catalogues
    echo "ssh "${SSH_USERNAME}@${SSH_HOST}" rm -rf ${SSH_PATH}/*"
    ssh "${SSH_USERNAME}@${SSH_HOST}" rm -rf ${SSH_PATH}/*
    # Copying over files
    echo "==> rsync -chavzP --stats "${DIR_LOCAL}/data/processed/TAR_files/"  "${CATL_COPY_COMMAND}"\n"
    rsync -chavzP --stats "${DIR_LOCAL}/data/processed/TAR_files/"  "${CATL_COPY_COMMAND}"
fi