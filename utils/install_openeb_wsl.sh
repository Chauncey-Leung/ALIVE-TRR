#!/usr/bin/env bash
# OpenEB RAW-only build for Ubuntu 20.04 x86_64 / Python 3.8.
set -eo pipefail
source /etc/os-release
if [[ "$ID" != ubuntu || "$VERSION_ID" != 20.04 || "$(uname -m)" != x86_64 ]]; then
    echo 'This recipe is verified only for Ubuntu 20.04 x86_64.' >&2
    exit 1
fi
if [[ "${1:-}" != --skip-system-deps ]]; then
    sudo apt-get update
    sudo apt-get install -y --no-install-recommends build-essential git cmake curl \
        libopencv-dev libboost-all-dev libusb-1.0-0-dev libhdf5-dev \
        python3.8-dev python3.8-venv
fi

OPENEB_DIR="$HOME/opt/openeb-4.6.2"
if [[ ! -d "$OPENEB_DIR/.git" ]]; then
    mkdir -p "$HOME/opt"
    git clone --depth 1 --branch 4.6.2 --recurse-submodules \
        https://github.com/prophesee-ai/openeb.git "$OPENEB_DIR"
fi
test "$(git -C "$OPENEB_DIR" rev-parse HEAD)" = 53b3618935f90dcc0f64993ccbb79514384404b0
python3.8 -m venv "$OPENEB_DIR/venv"
"$OPENEB_DIR/venv/bin/python" -m pip install \
    numpy==1.23.4 h5py==3.7.0 pybind11==2.6.0
cmake -S "$OPENEB_DIR" -B "$OPENEB_DIR/build" \
    -DBUILD_TESTING=OFF -DBUILD_SAMPLES=OFF -DCOMPILE_PYTHON3_BINDINGS=ON \
    '-DMETAVISION_SELECTED_MODULES=base;core' \
    -DPython3_EXECUTABLE="$OPENEB_DIR/venv/bin/python" \
    -Dpybind11_DIR="$OPENEB_DIR/venv/lib/python3.8/site-packages/pybind11/share/cmake/pybind11"
cmake --build "$OPENEB_DIR/build" --parallel 4
export PYTHONNOUSERSITE=1
source "$OPENEB_DIR/venv/bin/activate"
set --
source "$OPENEB_DIR/build/utils/scripts/setup_env.sh"
python -c 'from metavision_core.event_io.raw_reader import RawReader; print("RawReader import OK")'
