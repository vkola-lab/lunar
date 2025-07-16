# Training data for foundation model

---

## Order of running scripts to create NACC training and testing data

- Create patient history and diagnosis json files: `data_preparation.py` --- `data/{cohort}/training_data/{cohort}_wjson.csv`
- Select the last visit, add some neuropath cases to training set and save the final training and testing splits: `create_train_test_splits.ipynb` --- `data/{cohort}/training_data/train.csv`, `data/{cohort}/training_data/test.csv`
- **Training**
    - Create GRPO training questions: `create_grpo_training_data.ipynb` --- `data/{cohort}/training_data/training_data_grpo/train_with_questions.csv`.
    - Generate visit summaries --- `data/{cohort}/training_data/training_data_grpo/train_summary.csv`
        - Generate the summaries: `../create_summary/generate_patient_summary.sh`. 
        - Check the quality of the summaries: `../create_summary/check_generated_summaries.ipynb`. 
        - Repeat this procedure if required
    - Run inference on training cases to filter the cases for training --- `data/{cohort}/training_data/select_train_cases/train_selected.csv`.
        - generate 8 responses to each of the training prompts: `../../adrd_simplified_evaluation/run_benchmarks_train.sh`.
        - Extract the answers: `../../adrd_simplified_evaluation/lm_answer_extractor`.
        - Filter cases: `filter_grpo_data.ipynb`. Only prompts for which the base model answered at least once and less than 8 times is chosen for training.
    - Train using the final filtered cases.

- **Testing** (order of generating summary and creating questions reversed as the same summaries can be used to test all questions and they can be repeated)
    - Generate visit summaries --- `data/{cohort}/training_data/testing_data_grpo/test_summary.csv`
        - Generate the summaries: `../create_summary/generate_patient_summary.sh`. 
        - Check the quality of the summaries: `../create_summary/check_generated_summaries.ipynb`. 
        - Repeat this procedure if required.
    - Create GRPO training questions: `create_grpo_testing_data.ipynb` --- `data/{cohort}/training_data/testing_data_grpo/with_summary/`.