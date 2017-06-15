python -m bin.infer_raw \
  --tasks "
    - class: DecodeText" \
  --model_dir /home/chunfengh/nmt_data/toy_reverse/model/ \
  --batch_size 1 \
  --input_pipeline "
    class: ParallelTextInputPipeline
    params:
      source_files:
        - /home/chunfengh/nmt_data/toy_reverse/test/sources.txt" 

