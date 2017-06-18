python -m bin.infer \
  --tasks "
    - class: DecodeText" \
  --model_dir /home/chunfengh/nmt_data/toy_copy/model_conv \
  --batch_size 1 \
  --input_pipeline "
    class: ParallelTextInputPipeline
    params:
      source_files:
        - /home/chunfengh/nmt_data/toy_reverse/test/sources.txt" 

