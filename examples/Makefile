DATA_CONFIG := gene_expression/config.yaml
MODEL_CONFIG := gene_expression/config.yaml
TRAINING_CONFIG := gene_expression/config.yaml

OUTPUT_DIR := gene_expression/output

MODEL_OUTPUT := $(OUTPUT_DIR)/model.pth
HISTORY_OUTPUT := $(OUTPUT_DIR)/history.json

LOG_LEVEL := INFO

.PHONY: gene_expression

gene_expression:
	@mkdir -p $(OUTPUT_DIR)
	@uv run -m gene_expression.main \
	--data-config $(DATA_CONFIG) \
	--model-config $(MODEL_CONFIG) \
	--training-config $(TRAINING_CONFIG) \
	--model-output $(MODEL_OUTPUT) \
	--history-output $(HISTORY_OUTPUT) \
	--log-level $(LOG_LEVEL)