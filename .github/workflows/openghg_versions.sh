#!/usr/bin/env bash
# Script to get version numbers of OpenGHG
#
# The major version is always the most recent.
#
# You can specify a minor version, and the most recent release number for that minor version
# will be returned.
#
# Minor versions are specified as N behind the most recent.
#
# For instance, if the versions are: 1.2.1, 1.2.0, 1.1.0, 1.0.3, ...
#
# Then:
#
# ./openghg_version.sh -N 0
#
# returns 1.2.1,
#
# ./openghg_version.sh -N 1
#
# returns 1.1.0, and
#
# ./openghg_version.sh -N 2
#
# returns 1.0.3
#

minor_N=0
MAJOR_VERSION=-1
test=false

while getopts M:N:t flag
do
    case "${flag}" in
        t) test=true;;
        N) minor_N=${OPTARG};;
        M) MAJOR_VERSION=${OPTARG};;
    esac
done


# get version tags
OPENGHG_TAGS=$(curl https://api.github.com/repos/openghg/openghg/tags -s)
OPENGHG_VERSIONS_STR=$(echo $OPENGHG_TAGS | jq .[] | jq .name -r | grep -E "[0-9]+\.[0-9]+\.[0-9]+")

## NOTE: each version will be on a new line

# TODO: check major version

# if major version not specified, extract it.
# versions are sorted, so take major version from first entry.
if [[ $MAJOR_VERSION -eq -1 ]]; then
    MAJOR_VERSION=$(echo $OPENGHG_VERSIONS_STR | cut -d. -f1)
fi

# get minor versions for this major version
OPENGHG_VERSIONS_STR_MAJOR=$(echo $OPENGHG_VERSIONS_STR | grep -E "${MAJOR_VERSION}\.[0-9]+\.[0-9]+")
MINOR_VERSIONS=($(for x in $OPENGHG_VERSIONS_STR_MAJOR; do echo $x | cut -d. -f2; done | uniq))

# get specified minor
function get_minor()
{
    local MINOR_VERSION=${MINOR_VERSIONS[${1:0}]}  # $1 defaults to 0
    echo $OPENGHG_VERSIONS_STR | grep -oE "${MAJOR_VERSION}\.${MINOR_VERSION}\.[0-9]+" | head -n 1
}

# test or return
if [[ "$test" == true ]]; then
    echo "OPENGHG_VERSIONS_STR:"
    echo "${OPENGHG_VERSIONS_STR}"
    echo "MAJOR_VERSION = ${MAJOR_VERSION}"

    echo "MINOR_VERSIONS:"
    for x in ${MINOR_VERSIONS[*]}; do echo $x; done

    # get lastest release on last two minor versions
    ULTIMATE_MINOR_VERSION_LATEST=$(get_minor)
    PENULTIMATE_MINOR_VERSION_LATEST=$(get_minor 1)

    echo "Latest minor, most recent release = $ULTIMATE_MINOR_VERSION_LATEST"
    echo "Previous minor, most recent release = $PENULTIMATE_MINOR_VERSION_LATEST"
else
    # return specified minor
    get_minor $minor_N
fi
