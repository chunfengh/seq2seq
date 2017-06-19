python -m bin.train \
  --config_paths="
      ./example_configs/seq2inception.yml,
      ./example_configs/train_seq2seq.yml" \
  --batch_size 32 \
  --train_steps 500000 \
  --output_dir /home/chunfengh/nmt_data/toy_copy/model_inception/

