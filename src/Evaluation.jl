module Evaluation

using LinearAlgebra
using InlineStrings
using ..Word2VecModel
using ..Vocabulary

export cosine_similarity, get_vector, most_similar, analogy

"""
    cosine_similarity(x::AbstractVector, y::AbstractVector) -> Float64

Computes the cosine similarity between two vectors.
Adds epsilon to denominator to prevent division by zero.
"""
function cosine_similarity(x::AbstractVector, y::AbstractVector)
    ϵ = 1e-8  # small constant to avoid division by zero
    return dot(x, y) / (norm(x) * norm(y) + ϵ)
end



"""
    Word2Vec.get_vector(model::Word2Vec, vocab::Vocab, word::String)

Returns the embedding vector for a given word if it exists in the vocabulary.
"""
function get_vector(model::Word2Vec, vocab::Vocab, word::String)
    w = InlineString(word)
    if w ∉ keys(vocab.stoi)
        println("🚫 Word '$word' not found in vocabulary.")
        return nothing
    end
    return model.input_embeddings[:, vocab.stoi[w]]
end

"""
    Word2Vec.most_similar(model::Word2Vec, vocab::Vocab, query::String; topn=5)

Returns the top-N most similar words to a given query word using cosine similarity.
"""
function most_similar(model::Word2Vec, vocab::Vocab, query::String; topn=5)
    query_tok = InlineString(query)
    if query_tok ∉ keys(vocab.stoi)
        println("🚫 Word '$query' not found in vocabulary.")
        return nothing
    end

    query_vec = model.input_embeddings[:, vocab.stoi[query_tok]]
    sims = Dict{String, Float32}()

    for (word, idx) in vocab.stoi
        if word == query_tok
            continue
        end
        vec = model.input_embeddings[:, idx]
        sim = cosine_similarity(query_vec, vec)
        sims[String(word)] = sim
    end

    return sort(collect(sims), by = x -> -x[2])[1:topn]
end

"""
    Word2Vec.analogy(model::Word2Vec, vocab::Vocab, a::String, b::String, c::String; topn=1)

Solves word analogy questions of the form: a is to b as c is to ?
Computes: vec_b - vec_a + vec_c ≈ vec_d
"""
function analogy(model::Word2Vec, vocab::Vocab, a::String, b::String, c::String; topn=1)
    missing = [w for w in (a, b, c) if InlineString(w) ∉ keys(vocab.stoi)]
    if !isempty(missing)
        println("🚫 Missing words in vocabulary: ", join(missing, ", "))
        return []
    end

    vec_a = model.input_embeddings[:, vocab.stoi[InlineString(a)]]
    vec_b = model.input_embeddings[:, vocab.stoi[InlineString(b)]]
    vec_c = model.input_embeddings[:, vocab.stoi[InlineString(c)]]

    target_vec = vec_b - vec_a + vec_c

    sims = Dict{String, Float32}()
    for (word, idx) in vocab.stoi
        str_word = String(word)
        if str_word in (a, b, c)
            continue
        end
        vec = model.input_embeddings[:, idx]
        sim = cosine_similarity(target_vec, vec)
        sims[str_word] = sim
    end

    return sort(collect(sims), by = x -> -x[2])[1:topn]
end

end # module
