#!/usr/bin/env bash
#
# Ensure PyTensor has a non-empty compiler setting before starting pytest.

set -euo pipefail

readonly default_compiler_module="gcc/12.3.0-sknc"
readonly compiler_module="${PYTENSOR_COMPILER_MODULE:-${default_compiler_module}}"
readonly module_init="${PYTENSOR_MODULE_INIT:-/etc/profile.d/modules.sh}"
readonly bootstrap_mode="${PYTENSOR_COMPILER_BOOTSTRAP:-auto}"

pytensor_cxx() {
    python -c 'import pytensor; print(pytensor.config.cxx or "", end="")'
}

is_rocky_linux() {
    [[ -r /etc/os-release ]] || return 1

    (
        # shellcheck disable=SC1091
        source /etc/os-release
        [[ "${ID:-}" == "rocky" || " ${ID_LIKE:-} " == *" rocky "* ]]
    )
}

is_blue_pebble() {
    local host_name="${HOSTNAME:-}"

    if [[ -z "${host_name}" ]] && command -v hostname >/dev/null 2>&1; then
        host_name="$(hostname -s 2>/dev/null || true)"
    fi

    case "${host_name}" in
        [Bb][Pp]1* | *[Bb][Ll][Uu][Ee][Pp][Ee][Bb][Bb][Ll][Ee]* | *[Bb][Ll][Uu][Ee]-[Pp][Ee][Bb][Bb][Ll][Ee]*)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

load_compiler_module() {
    if ! type module >/dev/null 2>&1 && [[ -r "${module_init}" ]]; then
        # shellcheck disable=SC1090
        source "${module_init}"
    fi

    if ! type module >/dev/null 2>&1; then
        echo "PyTensor has no configured C++ compiler, and the environment-module command is unavailable." >&2
        echo "Set PYTENSOR_MODULE_INIT to the module init script, or set PYTENSOR_FLAGS=cxx=/path/to/g++." >&2
        return 1
    fi

    echo "PyTensor has no configured C++ compiler; loading module ${compiler_module}." >&2
    if ! module load "${compiler_module}"; then
        echo "Failed to load compiler module ${compiler_module}." >&2
        echo "Set PYTENSOR_COMPILER_MODULE to an available GCC module, or set PYTENSOR_FLAGS=cxx=/path/to/g++." >&2
        return 1
    fi
}

if [[ -z "$(pytensor_cxx)" ]]; then
    case "${bootstrap_mode}" in
        auto)
            if is_rocky_linux || is_blue_pebble; then
                load_compiler_module
            fi
            ;;
        always)
            load_compiler_module
            ;;
        never)
            ;;
        *)
            echo "Invalid PYTENSOR_COMPILER_BOOTSTRAP=${bootstrap_mode}; expected auto, always, or never." >&2
            exit 2
            ;;
    esac

    if [[ -z "$(pytensor_cxx)" ]]; then
        echo "PyTensor still has no configured C++ compiler; refusing to start pytest." >&2
        echo "Set PYTENSOR_FLAGS=cxx=/path/to/g++, or configure PYTENSOR_COMPILER_MODULE and PYTENSOR_MODULE_INIT." >&2
        echo "Set PYTENSOR_COMPILER_BOOTSTRAP=always to enable module loading on another host." >&2
        exit 2
    fi
fi

python -c 'import arviz'
exec "$@"
