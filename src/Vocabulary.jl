module Vocabulary

using InlineStrings
using ..Huffman

export Vocab, build_vocab

"""
    struct Vocab

A structure that holds vocabulary information for Word2Vec training.  
It includes word-to-index and index-to-word mappings, word frequencies, 
subsampling probabilities, and Huffman codes and paths for hierarchical softmax.
"""
struct Vocab
    stoi::Dict{InlineString, Int}                   # string-to-index mapping
    itos::Dict{Int, InlineString}                   # index-to-string mapping
    freqs::Dict{InlineString, Int}                  # word frequency counts
    subsampling_probs::Dict{InlineString, Float64}  # probability of dropping frequent words
    huffman_codes::Dict{InlineString, Vector{Int}}  # binary Huffman codes
    huffman_paths::Dict{InlineString, Vector{Int}}  # path as node indices
end

"""
    build_vocab(tokens::Union{Vector{InlineString}, Vector{Vector{InlineString}}}; min_freq=1) -> Vocab

Constructs a `Vocab` object from a list of tokens.

# Arguments
- `tokens`: 
    - Either a flat `Vector{InlineString}` of tokens, 
    - or a nested `Vector{Vector{InlineString}}` if tokens are grouped by sentences.
- `min_freq::Int=1`: Minimum frequency to include a word in the vocabulary.

# Returns
- A `Vocab` struct with mappings, frequencies, subsampling probabilities, and Huffman coding.

# Notes
- Automatically handles both flat and grouped token inputs.
- Frequent words can be filtered out using `min_freq`.
"""
function build_vocab(tokens::Union{Vector{InlineString}, Vector{Vector{InlineString}}}; min_freq::Int=1)::Vocab
    # 1. Flatten tokens if grouped by sentences
    flat_tokens = eltype(tokens) == InlineString ? tokens : vcat(tokens...)

    # 2. Count token frequencies
    freqs = Dict{InlineString, Int}()
    for token in flat_tokens
        freqs[token] = get(freqs, token, 0) + 1
    end

    # 3. Filter tokens by minimum frequency threshold
    filtered = filter(kv -> kv[2] ≥ min_freq, collect(freqs))

    # 4. Create token-to-index and index-to-token mappings
    stoi = Dict{InlineString, Int}()
    itos = Dict{Int, InlineString}()
    for (i, (word, _)) in enumerate(filtered)
        stoi[word] = i
        itos[i] = word
    end

    # 5. Compute subsampling probabilities based on word frequencies
    total = sum(kv[2] for kv in filtered)
    subsampling_probs = Dict{InlineString, Float64}()
    for (word, count) in filtered
        freq_ratio = count / total
        subsampling_probs[word] = 1 - sqrt(1e-5 / freq_ratio)
    end

    # 6. Huffman coding (if hierarchical softmax is used)
    filtered_freqs = Dict(filtered)
    huff_root = build_huffman_tree(filtered_freqs)
    codes, path_indices = generate_codes(huff_root)

    return Vocab(stoi, itos, freqs, subsampling_probs, codes, path_indices)
end

end # module
