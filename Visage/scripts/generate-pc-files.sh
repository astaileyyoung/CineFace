#!/bin/bash
set -e

mkdir -p /usr/local/lib/pkgconfig

VERSION_AVCODEC=$(basename $(ls /usr/local/lib/libavcodec.so.*.*.*) | cut -d'.' -f3-)
VERSION_AVFORMAT=$(basename $(ls /usr/local/lib/libavformat.so.*.*.*) | cut -d'.' -f3-)
VERSION_AVUTIL=$(basename $(ls /usr/local/lib/libavutil.so.*.*.*) | cut -d'.' -f3-)
VERSION_SWSCALE=$(basename $(ls /usr/local/lib/libswscale.so.*.*.*) | cut -d'.' -f3-)

cat > /usr/local/lib/pkgconfig/libavcodec.pc <<EOF
prefix=/usr/local
exec_prefix=\${prefix}
libdir=\${exec_prefix}/lib
includedir=\${prefix}/include

Name: libavcodec
Description: FFmpeg codec library
Version: $VERSION_AVCODEC
Libs: -L\${libdir} -lavcodec
Cflags: -I\${includedir}
EOF

cat > /usr/local/lib/pkgconfig/libavformat.pc <<EOF
prefix=/usr/local
exec_prefix=\${prefix}
libdir=\${exec_prefix}/lib
includedir=\${prefix}/include

Name: libavformat
Description: FFmpeg container format library
Version: $VERSION_AVFORMAT
Libs: -L\${libdir} -lavformat
Cflags: -I\${includedir}
EOF

cat > /usr/local/lib/pkgconfig/libavutil.pc <<EOF
prefix=/usr/local
exec_prefix=\${prefix}
libdir=\${exec_prefix}/lib
includedir=\${prefix}/include

Name: libavutil
Description: FFmpeg utility library
Version: $VERSION_AVUTIL
Libs: -L\${libdir} -lavutil
Cflags: -I\${includedir}
EOF

cat > /usr/local/lib/pkgconfig/libswscale.pc <<EOF
prefix=/usr/local
exec_prefix=\${prefix}
libdir=\${exec_prefix}/lib
includedir=\${prefix}/include

Name: libswscale
Description: FFmpeg image scaling library
Version: $VERSION_SWSCALE
Libs: -L\${libdir} -lswscale
Cflags: -I\${includedir}
EOF
