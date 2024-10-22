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

N=0
test=false

while getopts t:N: flag
do
    case "${flag}" in
        t) test=true;;
        N) N=${OPTARG};;
    esac
done


OPENGHG_VERSIONS_STR=$(curl https://api.github.com/repos/openghg/openghg/tags -s | jq .[] | jq .name -r | grep -E "\d+\.\d+\.\d+")

for x in $OPENGHG_VERSIONS_STR; do
    OPENGHG_VERSIONS+=($x)
done

# versions are sorted, so take first
MAJOR_VERSION=$(echo $OPENGHG_VERSIONS_STR | cut -d. -f1)

# get minor versions

## first minor version
MINOR_VERSIONS=($(echo $OPENGHG_VERSIONS_STR | cut -d. -f2))

## remaining minor versions
for x in $OPENGHG_VERSIONS_STR; do
    minor=$(echo $x | cut -d. -f2)
    major=$(echo $x | cut -d. -f1)
    if [[ $minor != ${MINOR_VERSIONS[-1]} && $major == $MAJOR_VERSION ]]; then
        MINOR_VERSIONS+=($minor)
    fi
done


# get lastest release on last two minor versions
ULTIMATE_MINOR_VERSION_LATEST=$(echo $OPENGHG_VERSIONS_STR | cut -d" " -f1)
PENULTIMATE_MINOR_VERSION_LATEST=$(echo $OPENGHG_VERSIONS_STR | grep -oE "${MAJOR_VERSION}\.${MINOR_VERSIONS[1]}\.\d+" | head -n 1)


# test
if [[ "$test" = true ]]; then
    echo "MAJOR_VERSION = ${MAJOR_VERSION}"

    echo "MINOR_VERSIONS:"
    for x in ${MINOR_VERSIONS[*]}; do echo $x; done

    echo "Latest minor, most recent release = $ULTIMATE_MINOR_VERSION_LATEST"
    echo "Previous minor, most recent release = $PENULTIMATE_MINOR_VERSION_LATEST"
fi

# return specified minor
MINOR_VERSION=${MINOR_VERSIONS[$N]}
echo $OPENGHG_VERSIONS_STR | grep -oE "${MAJOR_VERSION}\.${MINOR_VERSION}\.\d+" | head -n 1
