#!/bin/sh
# Installs Speakrail. Everything goes in ~/.speakrail, with its own Python, so nothing else on this machine changes.
#
#   curl -fsSL https://speakrail.com/install.sh | sh
#   curl -fsSL https://speakrail.com/install.sh | sh -s -- --yes
#
# --yes                no questions
# --home DIR           install there instead of ~/.speakrail (or set SPEAKRAIL_HOME)
# --models DIR         put the models there
# --config NAME        install this config instead of choosing one
# --no-modify-path     don't add speakrail to your shell's PATH
# --source DIR         install this checkout instead of downloading one
# SPEAKRAIL_BRANCH     the branch to download, main by default
set -eu

REPO="https://github.com/speakrail/speakrail.git"
UV_VERSION="0.11.8"

say() { printf '%s\n' "$*"; }
die() { printf '%s\n' "$*" >&2; exit 1; }

main() {
    home="${SPEAKRAIL_HOME:-$HOME/.speakrail}"
    source_dir=""
    yes=0
    pass=""
    while [ $# -gt 0 ]; do
        case "$1" in
            --yes|-y) yes=1; pass="$pass --yes"; shift ;;
            --no-modify-path) pass="$pass --no-modify-path"; shift ;;
            --home) [ $# -ge 2 ] || die "--home needs a folder"; home="$2"; shift 2 ;;
            --source) [ $# -ge 2 ] || die "--source needs a folder"; source_dir="$2"; shift 2 ;;
            --models|--config)
                [ $# -ge 2 ] || die "$1 needs a value"
                case "$2" in *[!A-Za-z0-9_./~-]*) die "$1: use a value without spaces or special characters" ;; esac
                pass="$pass $1 $2"; shift 2 ;;
            -h|--help) sed -n '2,13p' "$0" 2>/dev/null | sed 's/^# \{0,1\}//' || say "See https://speakrail.com"; exit 0 ;;
            *) die "Unknown option: $1" ;;
        esac
    done

    case "$(uname -s)-$(uname -m)" in
        Darwin-arm64) ;;
        Darwin-*) die "Speakrail needs Apple Silicon: an Intel Mac doesn't have the GPU memory it uses." ;;
        Linux-*) die "Linux and WSL aren't ready in this installer yet. They're coming with the NVIDIA engines." ;;
        *) die "Speakrail runs on Apple Silicon for now. On Windows it will come through WSL2." ;;
    esac
    if ! xcode-select -p >/dev/null 2>&1 || ! git --version >/dev/null 2>&1; then
        die "Speakrail compiles one part of itself, and that needs Apple's Command Line Tools. Install them with:

  xcode-select --install

then run this installer again."
    fi
    command -v curl >/dev/null 2>&1 || die "This installer needs curl."
    if [ "$yes" = 0 ] && ! (: </dev/tty) 2>/dev/null; then
        die "There's no terminal to ask questions in. Add --yes to go with what it suggests."
    fi

    fresh=""
    if [ ! -e "$home" ]; then fresh="--fresh"; fi
    mkdir -p "$home/bin"
    home="$(cd "$home" && pwd)"
    say "Installing to $home"

    # private to this folder: uv's cache and Python stay here, and uv leaves the shell alone
    export UV_CACHE_DIR="$home/cache/uv" UV_PYTHON_INSTALL_DIR="$home/python" UV_NO_MODIFY_PATH=1 UV_PYTHON_PREFERENCE=only-managed
    export SPEAKRAIL_HOME="$home"
    if [ ! -x "$home/bin/uv" ]; then
        say ""
        say "Downloading uv, which sets up Python..."
        curl -fsSL "https://astral.sh/uv/$UV_VERSION/install.sh" | env UV_UNMANAGED_INSTALL="$home/bin" sh >/dev/null 2>&1 \
            || die "Couldn't download uv, which sets up Python. Check your connection and run this again."
    fi
    uv="$home/bin/uv"
    say ""
    say "Getting Python 3.12..."
    "$uv" python install 3.12
    "$uv" -q venv --allow-existing --python 3.12 "$home/venv"

    say ""
    say "Getting Speakrail..."
    if [ -n "$source_dir" ]; then
        rsync -a --delete --exclude .git --exclude models --exclude run --exclude .venv --exclude __pycache__ "$source_dir/" "$home/app/"
    elif [ -d "$home/app/.git" ]; then
        git -C "$home/app" fetch -q --depth 1 origin "${SPEAKRAIL_BRANCH:-main}"
        git -C "$home/app" checkout -q FETCH_HEAD
    else
        git clone -q --depth 1 --branch "${SPEAKRAIL_BRANCH:-main}" "$REPO" "$home/app"
    fi
    say ""
    say "Installing its Python libraries..."
    "$uv" pip install --python "$home/venv/bin/python" -e "$home/app[install]"

    # shellcheck disable=SC2086  # $pass is words this script checked above
    exec "$home/venv/bin/python" -m speakrail install $fresh $pass </dev/null
}

main "$@"
