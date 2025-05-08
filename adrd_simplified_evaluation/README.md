# Notes
The results directory also contains a copy of the config file to ensure reproducibility

- create venv if necesary
- test that the benchmarks are in the correct format
- test that cuda is available 
- should we assume the user is logged in to hf?
- test that n_gpus is leq than the number we requested? torch will complain anyway
- we could set the generation up to ouput the token logits if we want to do an analysis on that 

# Tasks

- [X] make LoRA work
- [X] make multi GPU work
- [X] Each benchmark should have a JSON schema that specifies the answer structure
- [X] make the script such that you can override config arguments from command line. This helps submitting the jobs
- [ ] make the model answer directly without explanation
- [ ] try more benchmarks