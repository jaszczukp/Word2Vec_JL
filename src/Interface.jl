module Interface

using ..Tokenizer
using ..Vocabulary
using ..DataLoader
using ..Trainer
using ..Word2VecModel
using Serialization
using InlineStrings

export train!, predict, save_model, load_model


"""
    train!(model::Word2Vec, text::String; kwargs...) -> Nothing

Trains a `Word2Vec` model directly on raw text. This function performs
the full pipeline: tokenization, vocabulary construction, training data generation,
and model training using mini-batch gradient descent.

# Arguments
- `model::Word2Vec`: An instance of the Word2Vec model (CBOW or Skip-Gram).
- `text::String`: Raw input text for training.

# Keyword Arguments
- `window_size::Int=2`: Size of the context window (words before and after the target).
- `epochs::Int=5`: Number of training epochs.
- `lr::Float64=0.01`: Learning rate.
- `min_freq::Int=1`: Minimum word frequency to include in the vocabulary.
- `subsampling::Bool=true`: Whether to apply subsampling of frequent words.
- `batch_size::Int=32`: Number of training examples processed at once.
- `group_by_sentences::Bool=false`: Whether to split the text into sentences before tokenization.
- `verbose::Bool=true`: Whether to display training progress.

# Returns
- `Nothing`: The model is updated in place.
"""
function train!(model::Word2Vec, text::String;
                window_size::Int = 2,
                epochs::Int = 5,
                lr::Float64 = 0.01,
                min_freq::Int = 1,
                subsampling::Bool = true,
                batch_size::Int = 32,
                group_by_sentences::Bool = false,
                verbose::Bool = true)

    tokens = clean_and_tokenize(text; group_by_sentences=group_by_sentences)

    # Flatten tokens for vocabulary construction if split by sentences
    flat_tokens = eltype(tokens) == InlineString ? tokens : reduce(vcat, tokens)

    vocab = build_vocab(flat_tokens; min_freq=min_freq)
    data = generate_training_data(tokens, vocab; 
                                  window_size=window_size, 
                                  mode=model.mode, 
                                  subsampling=subsampling)

    model.vocab = vocab
    model.vocab_size = length(vocab.stoi)

    if isnothing(model.input_embeddings)
        model.input_embeddings = randn(Float32, model.embedding_dim, model.vocab_size)
        model.output_embeddings = randn(Float32, model.embedding_dim, model.vocab_size)
    end

    train_mode!(model, data, vocab; 
                 epochs=epochs, lr=lr, 
                 batch_size=batch_size, 
                 verbose=verbose)

    return nothing
end


"""
    predict_cbow(model::Word2Vec, context::Vector{String}) -> Union{String, Nothing}

Given a list of context words, predicts the most likely center word (target) using a CBOW model.

This function:
- Maps each context word to its index in the vocabulary.
- Computes the average of the corresponding input embeddings.
- Scores all vocabulary words against that average using dot product.
- Returns the word with the highest score as the prediction.

# Arguments
- `model::Word2Vec`: A trained Word2Vec model in `:cbow` mode.
- `context::Vector{String}`: List of surrounding context words.

# Returns
- `String`: The predicted center word.
- `nothing`: If none of the context words are found in the vocabulary or model is untrained.
"""
function predict_cbow(model::Word2Vec, context::Vector{String})
    if isnothing(model.vocab)
        error("🚫 Model must be trained before calling predict_cbow.")
    end

    # Convert context words to indices (if found in vocab)
    ctx_ids = [model.vocab.stoi[InlineString(w)] for w in context if haskey(model.vocab.stoi, InlineString(w))]
    if isempty(ctx_ids)
        return nothing  # No usable context
    end

    # Compute average input embedding for context
    input_vec = Word2VecModel.forward(model, ctx_ids)

    # Score all output embeddings against the input vector
    scores = model.output_embeddings' * input_vec |> vec

    # Find the word with the highest score
    pred_idx = argmax(scores)

    return String(model.vocab.itos[pred_idx])
end


"""
    predict_skipgram(model::Word2Vec, word::String; topn::Int = 5) -> Vector{String}

Given a single input word, returns the top-N most likely context words
based on similarity to its embedding. This simulates Skip-Gram behavior.

This function:
- Looks up the input word in the vocabulary.
- Fetches its input embedding.
- Computes dot-product scores against all output embeddings.
- Returns the top `topn` most similar words.

# Arguments
- `model::Word2Vec`: A trained Word2Vec model in `:skipgram` mode.
- `word::String`: Input word for which context is predicted.

# Keyword Arguments
- `topn::Int=5`: Number of predicted context words to return.

# Returns
- `Vector{String}`: List of predicted context words (most similar to the input).
- `[]`: If the word is not in the vocabulary or model is untrained.
"""
function predict_skipgram(model::Word2Vec, word::String; topn::Int = 5)
    if model.mode != :skipgram
        error("Model is not in Skip-Gram mode.")
    end
    if isnothing(model.vocab)
        error("Model must be trained before prediction.")
    end

    token = InlineString(word)
    if !haskey(model.vocab.stoi, token)
        println("Word '$word' not in vocabulary.")
        return []
    end

    input_idx = model.vocab.stoi[token]
    input_vec = model.input_embeddings[:, input_idx]
    scores = model.output_embeddings' * input_vec
    top_indices = partialsortperm(vec(scores), rev=true, 1:topn)

    return [String(model.vocab.itos[i]) for i in top_indices]
end

"""
    predict(model::Word2Vec, input::Union{Vector{String}, String}; topn::Int = 5) -> Union{String, Vector{String}}

General-purpose prediction function. Chooses between CBOW or Skip-Gram based on model.mode.

# Arguments
- `model::Word2Vec`: Trained Word2Vec model.
- `input`: 
    - For CBOW: a `Vector{String}` representing context words.
    - For Skip-Gram: a `String` representing a center word.
- `topn::Int=5`: Number of results to return (only for Skip-Gram).

# Returns
- CBOW: predicted target word (String)
- Skip-Gram: list of most likely context words (Vector{String})
"""
function predict(model::Word2Vec, input::Union{Vector{String}, String}; topn::Int = 5)
    if isnothing(model.vocab)
        error("🚫 Model must be trained before prediction.")
    end

    if model.mode == :cbow
        if !(input isa Vector{String})
            error("CBOW prediction expects a Vector{String} as input context.")
        end
        return predict_cbow(model, input)
    elseif model.mode == :skipgram
        if !(input isa String)
            error("Skip-Gram prediction expects a String as input center word.")
        end
        return predict_skipgram(model, input; topn=topn)
    else
        error("Unknown model mode: $(model.mode). Supported: :cbow or :skipgram.")
    end
end



"""
    save_model(model::Word2Vec, path::String) -> Nothing

Saves the model to a file using Julia's built-in `Serialization`.

# Arguments
- `model::Word2Vec`: Trained model.
- `path::String`: Output path for saving.

# Returns
Nothing.
"""
function save_model(model::Word2VecModel.Word2Vec, path::String)
    open(path, "w") do io
        serialize(io, model)
    end
end

"""
    load_model(path::String) -> Word2Vec

Loads a previously saved model.

# Arguments
- `path::String`: Path to a serialized model file.

# Returns
- `Word2Vec`: Deserialized model with vocabulary and weights.
"""
function load_model(path::String)::Word2VecModel.Word2Vec
    open(path, "r") do io
        return deserialize(io)
    end
end

end # module


