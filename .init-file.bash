export PS1="\h:\W \u\$ "
# Assume that this is set by running anaconda-project run shell, micromamba, etc.
# However, first allow the user (our Makefile) to explicitly set it by optionally
# defining CUSTOM_CONDA_PREFIX.  This allows one to avoid the possibility that
# the system files change this between invocation and loading of this script.
# See Issue #1.
CONDA_PREFIX="${CUSTOM_CONDA_PREFIX:-${CONDA_PREFIX}}"
CONDA_ENV="${CONDA_PREFIX}"
eval "$(micromamba shell hook --shell=bash)"

# [[ -v CONDA_PREFIX ]] is better, but needs bash > 4.2... not on Mac OS X.
while [[ ! -z ${CONDA_PREFIX+x} ]]; do 
    micromamba deactivate
done
micromamba activate "${CONDA_ENV}"

_exclude_array=(
    -name ".hg" -or
    -name ".git" -or
    -name '.eggs' -or
    -name '.ipynb_checkpoints' -or
    -name 'envs' -or 
    -name "*.sage-*" -or
    -name "_build" -or
    -name '_trash' -or
    -name '_oven' -or
    -name "build" -or
    -name "__pycache__"
)
# Finding stuff
function finda {
    find . \( "${_exclude_array[@]}" \) -prune -or -type f \
         -print0 | xargs -0 grep -H "${*:1}"
}

function findf {
    include_array=( -name "*.$1" )
    find . \( "${_exclude_array[@]}" \) -prune -or \( "${include_array[@]}" \) \
         -print0 | xargs -0 grep -H "${*:2}"
}

function findpy { findf py "${*}"; }
function findipy { findf ipynb "${*}"; }
function findjs { findf js "${*}"; }
function findcss { findf css "${*}"; }

if [[ $(uname -s) == Darwin ]]; then
    # Write this as a function so that we can run it in the background
    # Should make these use existing variables
    EMACSNAME=${EMACSNAME:-"Emacs"}
    EMACSAPP=${EMACSAPP:-"/Applications/${EMACSNAME}.app"}
    export EMACS=${EMACS:-"/Applications/${EMACSNAME}.app/Contents/MacOS/${EMACSNAME}"}
    export EMACSCLIENT=${EMACSCLIENT:-"${EMACSAPP}/Contents/MacOS/bin/emacsclient"}

    alias emacs_="${EMACS}"

    function emacs {
        "${EMACSCLIENT}" -a ${EMACS} --quiet --no-wait "$@" & disown
    }
    export emacs
fi
