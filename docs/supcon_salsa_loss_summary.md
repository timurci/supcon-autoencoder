### Contrastive Learning Loss: General Overview

Contrastive learning is a representation learning paradigm that aims to learn embeddings (vector representations) where similar data points are pulled closer together in the embedding space, while dissimilar points are pushed apart. This is typically achieved through a loss function that maximizes the similarity (often measured via dot product or cosine similarity) between "positive" pairs (similar samples) and minimizes it for "negative" pairs (dissimilar samples). A common formulation is inspired by noise contrastive estimation and involves a softmax-based term normalized by a temperature parameter (τ) to control the sharpness of the distribution.

The base contrastive loss for an anchor embedding \( z_i \) is often structured as:

- Numerator: Similarity to positives (e.g., \(\exp(z_i \cdot z_p / \tau)\) for a positive \( z_p \)).

- Denominator: Sum of similarities to all samples in the batch (positives + negatives).

This encourages the model to discriminate signals (positives) from noise (negatives). In self-supervised settings, positives are often augmentations of the same sample, while negatives are other samples in the batch. Supervised variants incorporate labels or domain-specific knowledge to define positives more richly.

### Application in the Original SupCon Paper (Khosla et al., 2021)

The Supervised Contrastive (SupCon) paper extends self-supervised contrastive learning to the fully supervised setting by leveraging class labels to define multiple positives per anchor, rather than relying solely on data augmentations. This creates "clusters" in the embedding space where samples from the same class are pulled together, while clusters from different classes are pushed apart. The authors analyze two variants of the supervised contrastive loss and identify the superior one (referred to as \( L^{sup}_{out} \)) based on empirical performance and gradient analysis.

#### Key Formulation

The SupCon loss operates on a "multiviewed batch": For a batch of \( N \) samples with labels, two random augmentations ("views") are applied to each, creating \( 2N \) augmented samples. Let:

- \( I = \{1, \dots, 2N\} \) be the indices of the multiviewed batch.

- \( z_\ell \) be the normalized projection of the \(\ell\)-th augmented sample (output of a projection network after the encoder).

- \( A(i) = I \setminus \{i\} \) (all indices except the anchor \( i \)).

- \( P(i) = \{ p \in A(i) : \tilde{y}_p = \tilde{y}_i \} \) (indices of positives: other views in the batch with the same label as \( i \), excluding \( i \) itself; cardinality \( |P(i)| \)).

- \( \tau > 0 \) be the temperature scalar.

The self-supervised baseline (for reference) is:

\[

L^{self} = \sum_{i \in I} -\log \frac{\exp(z_i \cdot z_{j(i)} / \tau)}{\sum_{a \in A(i)} \exp(z_i \cdot z_a / \tau)},

\]

where \( j(i) \) is the single positive (the other augmentation of the same original sample).

The supervised SupCon loss (the preferred variant, \( L^{sup}_{out} \), Eq. 2 in the paper) generalizes this to multiple positives:

\[

L^{sup} = \sum_{i \in I} -\frac{1}{|P(i)|} \sum_{p \in P(i)} \log \frac{\exp(z_i \cdot z_p / \tau)}{\sum_{a \in A(i)} \exp(z_i \cdot z_a / \tau)}.

\]

- This averages the log term over all positives in \( P(i) \), placed **outside** the log (hence "out").

- An alternative variant (\( L^{sup}_{in} \), Eq. 3) places the average **inside** the log, but it performs worse (e.g., 67.4% vs. 78.7% top-1 accuracy on ImageNet with ResNet-50) due to biased gradients and lack of normalization effects on positives.

#### How It's Applied

- **Architecture**: An encoder (e.g., ResNet-50/101/200) produces representations, followed by a projection network (MLP or linear layer) to get \( z \) (discarded at inference). Training is two-stage: (1) contrastive pretraining on the multiviewed batch, (2) fine-tuning a linear classifier on frozen representations with cross-entropy.

- **Data Augmentations**: Strong augmentations (e.g., AutoAugment, RandAugment) create views, ensuring positives include both augmentations and same-class samples.

- **Benefits and Results**: Encourages learning from hard positives/negatives (via gradient structure) without explicit mining. On ImageNet, it achieves state-of-the-art top-1 accuracy (e.g., 81.4% on ResNet-200, +0.8% over cross-entropy). It's more robust to corruptions (ImageNet-C), stable to hyperparameters (e.g., batch size, optimizer), and generalizes better on CIFAR-10/100.

- **Analytical Insights**: The loss subsumes triplet loss (1 positive, 1 negative) and N-pairs loss (1 positive, many negatives). Gradients intrinsically weigh hard examples more, improving clustering.

This is applied to image classification, showing consistent gains over cross-entropy and self-supervised methods.

### Application in the SALSA Paper (Kirchoff et al., 2023)

The SALSA (Semantically-Aware Latent Space Autoencoder) paper adapts the SupCon loss to molecular data represented as SMILES strings (text sequences for chemical graphs). The goal is to learn a latent space that respects "semantics" defined by structural similarity (graph-to-graph distance, specifically graph edit distance or GED). Unlike SupCon's class labels, positives are defined as "mutants" (molecules 1 GED away from an "anchor" molecule), enforcing that structurally similar molecules map to nearby latent codes. SALSA combines this contrastive loss with a reconstruction loss in a transformer autoencoder framework.

#### Key Formulation

SALSA uses the exact SupCon loss (matching \( L^{sup}_{out} \) from the original paper) as its contrastive component \( L_c \):

\[

L_c = \sum_{i \in I} -\frac{1}{|P(i)|} \sum_{p \in P(i)} \log \frac{\exp(z_i \cdot z_p / \tau)}{\sum_{a \in A(i)} \exp(z_i \cdot z_a / \tau)},

\]

- \( z_i \) is the normalized latent code (fixed-size vector from the autoencoder's bottleneck/pooling layer).

- \( I \) is the set of anchors in the batch.

- \( P(i) \) is the set of positives: 10 mutants per anchor \( i \) (1-GED graphs via node add/replace/remove operations).

- \( A(i) \) is the batch minus \( i \) (includes positives for \( i \) and negatives from other anchors/mutants).

- \( \tau = 0.7 \) (default).

The full loss is a weighted combination:

\[

\mathcal{L} = \lambda L_c + (1 - \lambda) L_r,

\]

where \( L_r \) is the reconstruction loss (causal cross-entropy on SMILES sequences), and \( \lambda = 0.5 \). This balances structural awareness (from \( L_c \)) with distinct representations (from \( L_r \), preventing collapse).

#### How It's Applied

- **Architecture**: Transformer encoder-decoder with a bottleneck (pooling to \( \mathbb{R}^S \), \( S=32 \)) and upsampling. Shared weights process anchors and mutants. The contrastive loss is on latent codes; reconstruction is on SMILES.

- **Dataset Generation**: Anchors from ChEMBL (~1.26M curated SMILES). Mutants generated via 1-GED operations (node add/replace/remove), filtered for physicochemical proximity (Mahalanobis distance threshold) and validity (RDKit). Batch: Multi-mutant (anchors + their 10 mutants each); positives are an anchor's own mutants, negatives are others.

- **Benefits and Results**: Creates a "semantically continuous" latent space. Evaluations show higher correlation between latent EuD and GED (e.g., Spearman's ρ: 0.578 for SALSA vs. 0.351 for naive autoencoder). Better interpolations (smooth property changes) and property awareness (e.g., encodes physicochemical traits like lipophilicity). Ablations confirm the combination outperforms contrastive-only (collapses) or reconstruction-only (disorganized).

- **Domain-Specific Twist**: Positives based on GED (structural edits) rather than labels, tailoring to chemistry. Faulty positives filtered to ensure semantic relevance. Applied to drug discovery tasks like property prediction and de novo generation.

In summary, both papers use the same core SupCon loss to handle multiple positives, but SupCon applies it to labeled image classification for better generalization/robustness, while SALSA adapts it to unlabeled molecular graphs (with generated positives) for a structurally aware autoencoder latent space.
