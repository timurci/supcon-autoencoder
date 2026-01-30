DATA_CONFIG := gene_expression/config.yaml
MODEL_CONFIG := gene_expression/config.yaml
TRAINING_CONFIG := gene_expression/config.yaml

OUTPUT_DIR := gene_expression/output

MODEL_OUTPUT := $(OUTPUT_DIR)/model.pth
HISTORY_OUTPUT := $(OUTPUT_DIR)/history.parquet

LOG_LEVEL := INFO

# Loss history plot parameters
LOSS_PLOT_INPUT := $(HISTORY_OUTPUT)
LOSS_PLOT_OUTPUT := $(OUTPUT_DIR)/loss_history.png
LOSS_PLOT_OFFSET := 0
LOSS_PLOT_TITLE := "Training and Validation Loss"
LOSS_PLOT_WIDTH := 10
LOSS_PLOT_HEIGHT := 6
LOSS_PLOT_DPI := 300

# Embedding plot parameters
EMBEDDING_DATA_CONFIG := $(DATA_CONFIG)
EMBEDDING_MODEL_PATH := $(MODEL_OUTPUT)
EMBEDDING_PROJECTION_OUTPUT := $(OUTPUT_DIR)/embedding_projections.png
EMBEDDING_DEVICE := cpu
EMBEDDING_RAND_SUBSET_SIZE :=
EMBEDDING_SEED :=

.PHONY: gene_expression loss_plot embedding_plot

all: gene_expression loss_plot embedding_plot

gene_expression:
	@mkdir -p $(OUTPUT_DIR)

	@cp $(DATA_CONFIG) $(OUTPUT_DIR)
	@cp $(MODEL_CONFIG) $(OUTPUT_DIR)
	@cp $(TRAINING_CONFIG) $(OUTPUT_DIR)

	@uv run -m gene_expression.main \
	--data-config $(DATA_CONFIG) \
	--model-config $(MODEL_CONFIG) \
	--training-config $(TRAINING_CONFIG) \
	--model-output $(MODEL_OUTPUT) \
	--history-output $(HISTORY_OUTPUT) \
	--log-level $(LOG_LEVEL)

loss_plot:
	@uv run -m gene_expression.loss_plot \
	--input $(LOSS_PLOT_INPUT) \
	--output $(LOSS_PLOT_OUTPUT) \
	--title $(LOSS_PLOT_TITLE) \
	--offset $(LOSS_PLOT_OFFSET) \
	--width $(LOSS_PLOT_WIDTH) \
	--height $(LOSS_PLOT_HEIGHT) \
	--dpi $(LOSS_PLOT_DPI)

embedding_plot:
	@uv run -m gene_expression.embedding_plot \
	--data-config $(EMBEDDING_DATA_CONFIG) \
	--model-path $(EMBEDDING_MODEL_PATH) \
	--device $(EMBEDDING_DEVICE) \
	$(if $(EMBEDDING_RAND_SUBSET_SIZE),--rand-subset-size $(EMBEDDING_RAND_SUBSET_SIZE)) \
	$(if $(EMBEDDING_SEED),--seed $(EMBEDDING_SEED)) \
	--projection-output $(EMBEDDING_PROJECTION_OUTPUT)