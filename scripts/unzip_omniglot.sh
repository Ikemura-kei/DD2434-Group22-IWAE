#! /bin/bash

mkdir -p data/omniglot

zip_list=(
        images_background_small1.zip
        images_background_small2.zip
        images_background.zip
        images_evaluation.zip
        strokes_background_small1.zip
        strokes_background_small2.zip
        strokes_background.zip
        strokes_evaluation.zip
        )

for zip_item in ${zip_list[@]}; do
  unzip "submodules/omniglot/python/$zip_item" -d data/omniglot
done

rm -r data/omniglot/__MACOSX