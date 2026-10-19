#!/bin/zsh
# Vendors the MiniMax-H3 engine (h3) from a checkout: copies its host
# directories (runtime/{engine,model,ops,media,...}) and Metal backend into
# runtime/video, its kernel groups into the video kernel group (GROUP/F ->
# video/GROUP_F, shared/F -> video/F), its kernel ABI into
# runtime/metal/abi/video and its C ABI header into runtime/video/abi, its
# converters into install/video, and rewrites their include paths for those
# locations (runtime/video/VENDORED). Then record the source commit in both
# VENDORED notes, rebuild and run make test-video-real.
#
#   dev/tools/vendor_video.sh H3_CHECKOUT
set -eu
SRC=${1:?usage: dev/tools/vendor_video.sh H3_CHECKOUT}
SRC=${SRC:A}
cd ${0:A:h:h:h}
hosts=()
for directory in $SRC/runtime/*(/); do
  [[ ${directory:t} == metal ]] || hosts+=(${directory:t})
done
for host in $hosts metal abi; do rm -rf runtime/video/$host; done
rm -rf runtime/metal/abi/video runtime/metal/kernels/video
mkdir -p runtime/video/{metal,abi} runtime/metal/abi/video runtime/metal/kernels/video/common
for host in $hosts; do cp -R $SRC/runtime/$host runtime/video/$host; done
cp $SRC/runtime/metal/MetalBackend.* runtime/video/metal/
cp $SRC/runtime/metal/abi/* runtime/metal/abi/video/
cp $SRC/runtime/metal/kernels/common/* runtime/metal/kernels/video/common/
for group in $SRC/runtime/metal/kernels/*(/); do
  case ${group:t} in
    common) ;;
    shared) cp $group/*.metal runtime/metal/kernels/video/ ;;
    *) for file in $group/*.metal; do cp $file runtime/metal/kernels/video/${group:t}_${file:t}; done ;;
  esac
done
cp $SRC/include/h3/h3.h runtime/video/abi/h3.h
cp $SRC/tools/convert_*.py install/video/
cp $SRC/h3/common.py $SRC/h3/config.py $SRC/h3/safetensors.py install/video/h3/
rewrite=()
for host in $hosts; do rewrite+=(-e "s#\"$host/#\"video/$host/#g"); done
files=($(find runtime/video/${^hosts} runtime/video/metal runtime/video/abi runtime/metal/abi/video \
  runtime/metal/kernels/video -type f \( -name '*.hpp' -o -name '*.h' -o -name '*.mm' -o -name '*.cpp' \
  -o -name '*.metal' \)))
sed -i '' -E $rewrite \
  -e 's#"metal/MetalBackend.hpp"#"video/metal/MetalBackend.hpp"#g' \
  -e 's#"metal/abi/#"metal/abi/video/#g' \
  -e 's#"metal/kernels/common/#"metal/kernels/video/common/#g' \
  -e 's#"h3/h3.h"#"video/abi/h3.h"#g' \
  $files
echo "vendored h3 $(git -C $SRC rev-parse HEAD) (host directories: $hosts)"
