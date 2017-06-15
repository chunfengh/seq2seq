python -m bin.train \
  --config_paths="
      ./example_configs/nmt_basic_zero.yml,
      ./example_configs/train_seq2seq.yml,
      ./example_configs/text_metrics_bpe.yml" \
  --batch_size 32 \
  --train_steps 100000 \
  --output_dir /home/chunfengh/nmt_data/toy_reverse/model_basic_zero/

