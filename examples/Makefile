# Gene expression example parameters
GENE_EXPRESSION_OUTPUT_DIR := gene_expression/output
GENE_EXPRESSION_DATA_CONFIG := gene_expression/config.yaml
GENE_EXPRESSION_MODEL_CONFIG := gene_expression/config.yaml
GENE_EXPRESSION_TRAINING_CONFIG := gene_expression/config.yaml

# Fashion MNIST example parameters
FASHION_MNIST_OUTPUT_DIR := fashion_mnist/output

# Gene expression embedding plot parameters
GENE_EXPRESSION_EMBEDDING_DEVICE := cpu
GENE_EXPRESSION_EMBEDDING_RAND_SUBSET_SIZE :=
GENE_EXPRESSION_EMBEDDING_SEED :=

# Fashion MNIST embedding plot parameters
FASHION_MNIST_EMBEDDING_DEVICE := cpu
FASHION_MNIST_EMBEDDING_RAND_SUBSET_SIZE :=
FASHION_MNIST_EMBEDDING_SEED :=

LOG_LEVEL := INFO

.PHONY: all gene_expression fashion_mnist train_gene_expression train_fashion_mnist embedding_plot_gene_expression embedding_plot_fashion_mnist clean

all: gene_expression fashion_mnist

# Full pipeline rules: train -> loss_plot -> embedding_plot
gene_expression: train_gene_expression embedding_plot_gene_expression

fashion_mnist: train_fashion_mnist embedding_plot_fashion_mnist

# Training rules
train_gene_expression:
	@mkdir -p $(GENE_EXPRESSION_OUTPUT_DIR)
	@cp $(GENE_EXPRESSION_DATA_CONFIG) $(GENE_EXPRESSION_OUTPUT_DIR)/config.data.yaml
	@cp $(GENE_EXPRESSION_MODEL_CONFIG) $(GENE_EXPRESSION_OUTPUT_DIR)/config.model.yaml
	@cp $(GENE_EXPRESSION_TRAINING_CONFIG) $(GENE_EXPRESSION_OUTPUT_DIR)/config.training.yaml
	@uv run -m gene_expression.main \
	--data-config $(GENE_EXPRESSION_DATA_CONFIG) \
	--model-config $(GENE_EXPRESSION_MODEL_CONFIG) \
	--training-config $(GENE_EXPRESSION_TRAINING_CONFIG) \
	--model-output $(GENE_EXPRESSION_OUTPUT_DIR)/model.pth \
	--log-level $(LOG_LEVEL)

train_fashion_mnist:
	@mkdir -p $(FASHION_MNIST_OUTPUT_DIR)
	@uv run -m fashion_mnist.main \
	--model-output $(FASHION_MNIST_OUTPUT_DIR)/fashion_mnist_autoencoder.pt

# Embedding plot rules
embedding_plot_gene_expression:
	@uv run -m gene_expression.embedding_plot \
	--data-config $(GENE_EXPRESSION_OUTPUT_DIR)/config.data.yaml \
	--model-path $(GENE_EXPRESSION_OUTPUT_DIR)/model.pth \
	--device $(GENE_EXPRESSION_EMBEDDING_DEVICE) \
	$(if $(GENE_EXPRESSION_EMBEDDING_RAND_SUBSET_SIZE),--rand-subset-size $(GENE_EXPRESSION_EMBEDDING_RAND_SUBSET_SIZE)) \
	$(if $(GENE_EXPRESSION_EMBEDDING_SEED),--seed $(GENE_EXPRESSION_EMBEDDING_SEED)) \
	--projection-output $(GENE_EXPRESSION_OUTPUT_DIR)/embedding_projections.png

embedding_plot_fashion_mnist:
	@uv run -m fashion_mnist.embedding_plot \
	--model-path $(FASHION_MNIST_OUTPUT_DIR)/fashion_mnist_autoencoder.pt \
	--projection-output $(FASHION_MNIST_OUTPUT_DIR)/embedding_projections.png

clean:
	rm -rf $(GENE_EXPRESSION_OUTPUT_DIR)
	rm -rf $(FASHION_MNIST_OUTPUT_DIR)
