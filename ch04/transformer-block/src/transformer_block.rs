use burn_helpers::attention_algorithms::MultiHeadAttention;

struct TransforerBlock {
    attention: MultiHeadAttention,
}

impl TransformerBlock {
    pub fn new(config: GPTConfig124M) -> Self {
        let attention = MultiHeadAttention::new(
            config.emb_dim,
            config.context_length,
            config.n_heads,
            config.dropout_rate,
        );
        Self { attention }
    }
}
