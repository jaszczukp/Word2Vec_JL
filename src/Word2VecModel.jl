module Word2VecModel

using ..Vocabulary

export Word2Vec, forward

"""
    mutable struct Word2Vec

Defines a Word2Vec model with input and output embeddings, the training mode (CBOW or Skip-Gram),
and the loss type (e.g. negative sampling, hierarchical softmax).
"""
mutable struct Word2Vec
    vocab_size::Union{Nothing, Int}                   # Number of words in vocabulary
    embedding_dim::Int                                # Dimensionality of embedding vectors
    input_embeddings::Union{Nothing, Matrix{Float32}} # Input embedding matrix (dim × vocab_size)
    output_embeddings::Union{Nothing, Matrix{Float32}}# Output embedding matrix (dim × vocab_size)
    mode::Symbol                                       # :cbow or :skipgram
    loss_type::Symbol                                  # :neg_sampling, or :hs
    vocab::Union{Nothing, Vocab}                       # Vocabulary (added during training)
end

"""
    Word2Vec(embedding_dim::Int;
             mode::Symbol = :cbow,
             loss_type::Symbol = :hs) -> Word2Vec

Initializes a new Word2Vec model with random weights and no vocabulary.
"""
function Word2Vec(embedding_dim::Int;
                  mode::Symbol = :cbow,
                  loss_type::Symbol = :hs)
    return Word2Vec(nothing, embedding_dim, nothing, nothing, mode, loss_type, nothing)
end

"""
    forward(model::Word2Vec, input_indices::Vector{Int}) -> Matrix{Float32}

Computes the forward pass of the model:
- For CBOW: average of input word embeddings.
- For Skip-Gram: single input embedding.

# Arguments
- `input_indices::Vector{Int}`: Indices of input/context words.

# Returns
- Matrix{Float32} of shape (embedding_dim, 1)
"""
function forward(model::Word2Vec, input_indices::Vector{Int})
    if model.mode == :cbow
        embeddings = model.input_embeddings[:, input_indices]
        return sum(embeddings; dims = 2) / length(input_indices)
    elseif model.mode == :skipgram
        @assert length(input_indices) == 1 "Skip-gram expects a single input index."
        return model.input_embeddings[:, input_indices[1]]
    else
        error("Unknown model mode: $(model.mode). Use :cbow or :skipgram.")
    end
end

end # module
