module Trainer

using ..Word2VecModel
using ..Vocabulary
using LinearAlgebra
using Random

export train_mode!

# ─────────────────────────────────────────────────────────────────────────────
# Logistic sigmoid activation function
σ(x) = 1 / (1 + exp(-x))

# ─────────────────────────────────────────────────────────────────────────────
"""
    sample_negative(vocab::Vocab, k::Int; skip::Int) -> Vector{Int}

Randomly samples `k` negative word indices from the vocabulary,
excluding the `skip` index (typically the true target).
"""
function sample_negative(vocab::Vocab, k::Int; skip::Int)
    indices = collect(1:length(vocab.stoi))
    filtered = filter(i -> i != skip, indices)
    return rand(filtered, k)
end

# ─────────────────────────────────────────────────────────────────────────────
function train_cbow_hierarchical_softmax!(model::Word2Vec, data, vocab::Vocab;
                                          epochs=5, lr=0.01, batch_size=32, verbose=true)
    for epoch in 1:epochs
        total_loss = 0.0
        shuffled_data = shuffle(data)

        for i in 1:batch_size:length(shuffled_data)
            batch = shuffled_data[i:min(i+batch_size-1, end)]
            grad_in = zeros(Float32, size(model.input_embeddings))
            grad_out = zeros(Float32, size(model.output_embeddings))
            batch_loss = 0.0

            for (context_indices, target_index) in batch
                input_vec = sum(model.input_embeddings[:, context_indices]; dims=2) / length(context_indices)
                target_word = vocab.itos[target_index]
                code = vocab.huffman_codes[target_word]
                path = vocab.huffman_paths[target_word]

                δ_input = zeros(Float32, size(input_vec))

                for (bit, output_idx) in zip(code, path)
                    out_vec = model.output_embeddings[:, output_idx]
                    score = dot(input_vec, out_vec)
                    pred = σ(score)
                    error = pred - bit
                    loss = -log(bit == 1 ? clamp(pred, 1e-7, 1.0) : clamp(1 - pred, 1e-7, 1.0))
                    batch_loss += loss

                    grad_out[:, output_idx] .+= error .* input_vec[:]
                    δ_input[:] .+= error .* out_vec
                end

                for idx in context_indices
                    grad_in[:, idx] .+= (δ_input[:] ./ length(context_indices))
                end
            end

            model.input_embeddings .-= lr .* grad_in
            model.output_embeddings .-= lr .* grad_out
            total_loss += batch_loss
        end

        avg_loss = total_loss / length(data)
        if verbose
            println("⚡ Epoch $epoch – cbow + hs (batch=$batch_size) loss: ", round(avg_loss, digits=5))
        end
    end
end

# ─────────────────────────────────────────────────────────────────────────────
function train_cbow_neg_sampling!(model::Word2Vec, data, vocab::Vocab;
                                  epochs=5, lr=0.01, k=5, batch_size=32, verbose=true)
    for epoch in 1:epochs
        total_loss = 0.0
        shuffled_data = shuffle(data)

        for i in 1:batch_size:length(shuffled_data)
            batch = shuffled_data[i:min(i+batch_size-1, end)]
            grad_in = zeros(Float32, size(model.input_embeddings))
            grad_out = zeros(Float32, size(model.output_embeddings))
            batch_loss = 0.0

            for (context_indices, target_idx) in batch
                input_vec = sum(model.input_embeddings[:, context_indices]; dims=2) / length(context_indices)
                target_vec = model.output_embeddings[:, target_idx]

                score = dot(input_vec, target_vec)
                pred = σ(score)
                loss = -log(clamp(pred, 1e-7, 1.0))
                batch_loss += loss

                grad = pred - 1
                grad_out[:, target_idx] .+= grad .* input_vec
                input_vec[:] .-= grad .* target_vec

                neg_samples = sample_negative(vocab, k; skip=target_idx)
                for neg_idx in neg_samples
                    neg_vec = model.output_embeddings[:, neg_idx]
                    score = dot(input_vec, neg_vec)
                    pred = σ(score)
                    loss = -log(clamp(1 - pred, 1e-7, 1.0))
                    batch_loss += loss

                    grad = pred
                    grad_out[:, neg_idx] .+= grad .* input_vec
                    input_vec[:] .-= grad .* neg_vec
                end

                for idx in context_indices
                    grad_in[:, idx] .+= (input_vec[:] ./ length(context_indices))
                end
            end

            model.input_embeddings .-= lr .* grad_in
            model.output_embeddings .-= lr .* grad_out
            total_loss += batch_loss
        end

        avg_loss = total_loss / length(data)
        if verbose
            println("⚡ Epoch $epoch – cbow + ns (batch=$batch_size) loss: ", round(avg_loss, digits=5))
        end
    end
end

# ─────────────────────────────────────────────────────────────────────────────
function train_skipgram_hier_softmax!(model::Word2Vec, data, vocab::Vocab;
                                      epochs=5, lr=0.01, batch_size=32, verbose=true)
    for epoch in 1:epochs
        total_loss = 0.0
        shuffled_data = shuffle(data)

        for i in 1:batch_size:length(shuffled_data)
            batch = shuffled_data[i:min(i+batch_size-1, end)]
            grad_in = zeros(Float32, size(model.input_embeddings))
            grad_out = zeros(Float32, size(model.output_embeddings))
            batch_loss = 0.0

            for (input_idx, target_idx) in batch
                input_vec = model.input_embeddings[:, input_idx]
                word = vocab.itos[target_idx]
                code = vocab.huffman_codes[word]
                path = vocab.huffman_paths[word]

                for (bit, output_idx) in zip(code, path)
                    out_vec = model.output_embeddings[:, output_idx]
                    score = dot(input_vec, out_vec)
                    pred = σ(score)
                    error = pred - bit
                    loss = -log(bit == 1 ? clamp(pred, 1e-7, 1.0) : clamp(1 - pred, 1e-7, 1.0))
                    batch_loss += loss

                    grad_out[:, output_idx] .+= error .* input_vec
                    grad_in[:, input_idx] .+= error .* out_vec
                end
            end

            model.input_embeddings .-= lr .* grad_in
            model.output_embeddings .-= lr .* grad_out
            total_loss += batch_loss
        end

        avg_loss = total_loss / length(data)
        if verbose
            println("⚡ Epoch $epoch – skipgram + hs (batch=$batch_size) loss: ", round(avg_loss, digits=5))
        end
    end
end

# ─────────────────────────────────────────────────────────────────────────────
function train_skipgram_neg_sampling!(model::Word2Vec, data, vocab::Vocab;
                                      epochs=5, lr=0.01, k=5, batch_size=32, verbose=true)
    for epoch in 1:epochs
        total_loss = 0.0
        shuffled_data = shuffle(data)

        for i in 1:batch_size:length(shuffled_data)
            batch = shuffled_data[i:min(i+batch_size-1, end)]
            grad_in = zeros(Float32, size(model.input_embeddings))
            grad_out = zeros(Float32, size(model.output_embeddings))
            batch_loss = 0.0

            for (input_idx, target_idx) in batch
                input_vec = model.input_embeddings[:, input_idx]
                target_vec = model.output_embeddings[:, target_idx]

                score = dot(input_vec, target_vec)
                pred = σ(score)
                loss = -log(clamp(pred, 1e-7, 1.0))
                batch_loss += loss

                grad = pred - 1
                grad_out[:, target_idx] .+= grad .* input_vec
                grad_in[:, input_idx] .+= grad .* target_vec

                neg_samples = sample_negative(vocab, k; skip=target_idx)
                for neg_idx in neg_samples
                    neg_vec = model.output_embeddings[:, neg_idx]
                    score = dot(input_vec, neg_vec)
                    pred = σ(score)
                    loss = -log(clamp(1 - pred, 1e-7, 1.0))
                    batch_loss += loss

                    grad = pred
                    grad_out[:, neg_idx] .+= grad .* input_vec
                    grad_in[:, input_idx] .+= grad .* neg_vec
                end
            end

            model.input_embeddings .-= lr .* grad_in
            model.output_embeddings .-= lr .* grad_out
            total_loss += batch_loss
        end

        avg_loss = total_loss / length(data)
        if verbose
            println("⚡ Epoch $epoch – skipgram + ns (batch=$batch_size) loss: ", round(avg_loss, digits=5))
        end
    end
end

# ─────────────────────────────────────────────────────────────────────────────
function train_mode!(model::Word2Vec, data, vocab::Vocab;
                     epochs=5, lr=0.01, batch_size=32, verbose=true)
    if model.mode == :cbow && model.loss_type == :hs
        train_cbow_hierarchical_softmax!(model, data, vocab; epochs, lr, batch_size, verbose)
    elseif model.mode == :cbow && model.loss_type == :neg_sampling
        train_cbow_neg_sampling!(model, data, vocab; epochs, lr, k=3, batch_size, verbose)
    elseif model.mode == :skipgram && model.loss_type == :hs
        train_skipgram_hier_softmax!(model, data, vocab; epochs, lr, batch_size, verbose)
    elseif model.mode == :skipgram && model.loss_type == :neg_sampling
        train_skipgram_neg_sampling!(model, data, vocab; epochs, lr, k=3, batch_size, verbose)
    else
        error("Unsupported mode: mode=$(model.mode), loss_type=$(model.loss_type)")
    end
end

end # module
