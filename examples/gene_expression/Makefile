DATA_CONFIG := config.yaml
MODEL_CONFIG := config.yaml
TRAINING_CONFIG := config.yaml

MODEL_OUTPUT := model.pth
HISTORY_OUTPUT := history.json

all: train

train:
	@uv run -m main \
	--data-config $(DATA_CONFIG) \
	--model-config $(MODEL_CONFIG) \
	--training-config $(TRAINING_CONFIG) \
	--model-output $(MODEL_OUTPUT) \
	--history-output $(HISTORY_OUTPUT)