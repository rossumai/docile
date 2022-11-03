#!/bin/bash

for source_target_pair in "glib/lib/libgobject-2.0.0.dylib gobject-2.0" "pango/lib/libpango-1.0.dylib pango-1.0" "harfbuzz/lib/libharfbuzz.dylib harfbuzz" "fontconfig/lib/libfontconfig.1.dylib fontconfig-1" "pango/lib/libpangoft2-1.0.dylib pangoft2-1.0"; do
  set -- ${source_target_pair}
  source_path=/opt/homebrew/opt/$1
  target_path=/usr/local/lib/$2
  if [ ! -f ${target_path} ]; then
    ln -s ${source_path} ${target_path}
  fi
done
