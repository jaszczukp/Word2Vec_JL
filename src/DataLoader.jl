module DataLoader

using InlineStrings

export generate_training_data

"""
    generate_training_data(tokens, vocab;
                           window_size::Int=2,
                           mode::Symbol=:cbow,
                           subsampling::Bool=true)

Generates training pairs for the Word2Vec model.

# Arguments
- `tokens`: 
    - Can be a `Vector{InlineString}` (a flat list of tokens),
    - or a `Vector{Vector{InlineString}}` (tokenized by sentences).
- `vocab`: Vocabulary object containing mappings and subsampling information.
- `window_size::Int=2`: Context window size (number of words before and after the target).
- `mode::Symbol=:cbow`: Either `:cbow` or `:skipgram`.
- `subsampling::Bool=true`: Whether to apply subsampling to frequent words.

# Returns
- `Vector{Tuple}`: A list of training pairs (input, target).
"""
function generate_training_data(tokens,
                                vocab;
                                window_size::Int=2,
                                mode::Symbol=:cbow,
                                subsampling::Bool=true)

    pairs = []

    if eltype(tokens) == InlineString
        sentence_batches = [tokens]  
    elseif eltype(tokens) <: AbstractVector
        sentence_batches = tokens  
    else
        error("Unsupported tokens format. Provide either Vector{InlineString} or Vector{Vector{InlineString}}.")
    end

    # Processing of each sentence
    for sentence in sentence_batches
        for i in eachindex(sentence)
            word = sentence[i]

            # Subsampling of frequent words
            if subsampling
                prob = clamp(get(vocab.subsampling_probs, word, 1.0), 0.0, 1.0)
                if rand() < prob
                    continue
                end
            end

            # Context window limited by sentence length
            left = max(first(eachindex(sentence)), i - window_size)
            right = min(last(eachindex(sentence)), i + window_size)
            context = [sentence[j] for j in left:right if j != i]

            # Subsampling of context words
            if subsampling
                context = [w for w in context if rand() ≤ get(vocab.subsampling_probs, w, 1.0)]
            end

            if isempty(context)
                continue
            end

            # Conversion of words to indices in the dictionary
            input_ids = [vocab.stoi[w] for w in context if haskey(vocab.stoi, w)]
            target_id = get(vocab.stoi, word, 0)

            if target_id == 0 || any(x -> x == 0, input_ids)
                continue
            end

            # Creating training pairs according to the selected mode
            if mode == :cbow
                push!(pairs, (input_ids, target_id))  # (context -> word)
            elseif mode == :skipgram
                for input in input_ids
                    push!(pairs, (target_id, input))  # (word -> each context word)
                end
            else
                error("Unknown mode: $mode. Use :cbow or :skipgram.")
            end
        end
    end

    return pairs
end

end # module
