#!/bin/zsh
# Vendors the MiniMax-H3 engine (h3) from a checkout: copies its engine,
# model, ops and Metal sources into runtime/video, its kernels into the video
# kernel group and its kernel ABI into runtime/metal/abi/video, and rewrites
# their include paths for those locations (runtime/video/VENDORED). Then
# record the source commit in VENDORED, rebuild and run make test-video.
#
#   dev/tools/vendor_video.sh H3_CHECKOUT
set -eu
SRC=${1:?usage: dev/tools/vendor_video.sh H3_CHECKOUT}
SRC=${SRC:A}
cd ${0:A:h:h:h}
rm -rf runtime/video/{engine,model,ops,metal,abi} runtime/metal/abi/video runtime/metal/kernels/video
mkdir -p runtime/video/{engine,model,ops,metal,abi} runtime/metal/abi/video runtime/metal/kernels/video/common
cp $SRC/runtime/engine/* runtime/video/engine/
cp $SRC/runtime/model/* runtime/video/model/
cp $SRC/runtime/ops/* runtime/video/ops/
cp $SRC/runtime/metal/MetalBackend.* runtime/video/metal/
cp $SRC/runtime/metal/abi/* runtime/metal/abi/video/
cp $SRC/runtime/metal/kernels/common/* runtime/metal/kernels/video/common/
for group in dit vae preview; do
  for file in $SRC/runtime/metal/kernels/$group/*.metal; do
    cp $file runtime/metal/kernels/video/${group}_${file:t}
  done
done
cp $SRC/runtime/metal/kernels/shared/*.metal runtime/metal/kernels/video/
cp $SRC/include/h3/h3.h runtime/video/abi/h3.h
files=($(find runtime/video/{engine,model,ops,metal,abi} runtime/metal/abi/video runtime/metal/kernels/video \
  -type f \( -name '*.hpp' -o -name '*.h' -o -name '*.mm' -o -name '*.cpp' -o -name '*.metal' \)))
sed -i '' -E \
  -e 's#"engine/#"video/engine/#g' \
  -e 's#"model/#"video/model/#g' \
  -e 's#"ops/#"video/ops/#g' \
  -e 's#"metal/MetalBackend.hpp"#"video/metal/MetalBackend.hpp"#g' \
  -e 's#"metal/abi/#"metal/abi/video/#g' \
  -e 's#"metal/kernels/common/#"metal/kernels/video/common/#g' \
  -e 's#"h3/h3.h"#"video/abi/h3.h"#g' \
  $files
echo "vendored h3 $(git -C $SRC rev-parse HEAD)"
