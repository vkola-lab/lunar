# Directories
CONFIG_DIR = configs
RESULTS_DIR = results/comparison_inc
FIGURES_DIR = figures_inc
LOGS_DIR = logs
STAMP_DIR = logs/timestamps

# Declare phony targets
.PHONY: all help run_benchmark submit_all extract_answers compute_metrics figures clean_logs clean_figures clean_metrics clean

# Default target: usage message
all:
	@echo "Run 'make help' for help."

help:
	@echo "Usage:"
	@echo "  make run_benchmark CONFIG=configs/my_config.yml   # run a single benchmark"
	@echo "  make submit_all CONFIG_DIR=configs/               # submit all benchmarks in a directory"
	@echo "  make extract_answers                              # extract answers from results"
	@echo "  make compute_metrics                              # compute metrics after extracting answers"
	@echo "  make figures                                      # generate all figures"
	@echo "  make clean                                        # remove logs"

# Run a single benchmark with a given config.yml
run_benchmark:
	qsub run_benchmark.sh $(CONFIG)
	touch $(STAMP_DIR)/run_benchmark.stamp

# Submit all benchmarks for a directory of configs
submit_all:
	./submit_all.sh $(CONFIG_DIR)
	touch $(STAMP_DIR)/run_benchmark.stamp

# Always run extract_answers.sh, produce a stamp file
extract_answers: $(STAMP_DIR)/run_benchmark.stamp
	qsub extract_answers.sh
	touch $(STAMP_DIR)/extract_answers.stamp

# Always run compute_metrics.sh, depends on extract_answers stamp
compute_metrics: $(STAMP_DIR)/extract_answers.stamp
	./compute_metrics.sh
	touch $(STAMP_DIR)/compute_metrics.stamp

# Meta-target to build all figures
# break over multiple lines with backslash, make sure there is no trailing whitespace
figures: $(FIGURES_DIR)/accuracy_mcq.pdf \
		$(FIGURES_DIR)/invalid_num.pdf \
		$(FIGURES_DIR)/invalid_frac.pdf \
		$(FIGURES_DIR)/nacc_macro_bars.pdf \
		$(FIGURES_DIR)/nacc_weighted_bars.pdf \
		$(STAMP_DIR)/class_metrics_figures.stamp

$(STAMP_DIR)/class_metrics_figures.stamp: $(STAMP_DIR)/compute_metrics.stamp 
	python src/figures/plot_nacc_class_metrics.py $(RESULTS_DIR) $(FIGURES_DIR)  
	touch $@

# Rule for generating the figures, the python script must start with plot
# The plotting scripts load all metrics available in RESULTS_DIR
$(FIGURES_DIR)/%.pdf: $(STAMP_DIR)/compute_metrics.stamp
	@mkdir -p $(FIGURES_DIR)
	python src/figures/plot_$*.py $(RESULTS_DIR) $@

clean:
	@echo "Choose what to clean:"
	@echo "make clean_logs     # delete logs in the logs directory, i.e. *.stderr and *.stdout" 
	@echo "make clean_figures  # delete all figures, i.e. figures/*.pdf" 
	@echo "make clean_metrics  # delete all metrics from the results directory, i.e. metrics.json" 

# Clean logs
clean_logs:
	@echo "Removing logs:"
	rm -f $(LOGS_DIR)/*.stdout $(LOGS_DIR)/*.stderr

# Clean figures
clean_figures:
	@echo "Removing figures:"
	rm -f $(FIGURES_DIR)/*.pdf
	rm -f $(STAMP_DIR)/class_metrics_figures.stamp

# The logic is not cleanly separated, the python file creating the metrics will check if they exist
# and skip if they do. This should probably be handled by make instead
clean_metrics:
	@echo "Removing computed metrics:"
	@find $(RESULTS_DIR) -type f \( -name "metrics.json" -o -name "class_metrics.json" \) -exec rm -f {} +
	@rm -f $(STAMP_DIR)/compute_metrics.stamp