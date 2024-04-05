## gen text
python data/gen_text.py \
-s examples/data \
-d examples/dataset

## detect object
python data/detect_obj.py \
-s examples/data \
-d examples/dataset

## get multi-views
python data/make_multiview.py \
-s examples/data \
-d examples/dataset