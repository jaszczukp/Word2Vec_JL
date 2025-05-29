using Test
using Word2Vec_JL
using Word2Vec_JL.Interface
using InlineStrings
using LinearAlgebra

@testset "Tokenizer" begin
    tokens = Word2Vec_JL.clean_and_tokenize("The quick brown fox jumps over the lazy dog.")
    @test length(tokens) == 9
    @test "the" ∈ tokens
    @test "fox" ∈ tokens
end

@testset "Vocabulary" begin
    tokens = Word2Vec_JL.clean_and_tokenize("the the the fox fox jumps")
    vocab = Word2Vec_JL.build_vocab(tokens)
    @test haskey(vocab.stoi, InlineString("fox"))
    @test vocab.freqs[InlineString("the")] == 3
    @test length(vocab.stoi) == 3
end

@testset "Model Training (CBOW + neg_sampling)" begin
    text = """
    the king and the queen ruled the kingdom.
    the princess and the prince were loved by the people.
    """
    model, vocab = Word2Vec_JL.Interface.train_word2vec(
        text;
        mode=:cbow,
        loss_type=:neg_sampling,
        dim=10,
        window_size=2,
        epochs=3,
        verbose=false
    )

    w1 = InlineString("king")
    w2 = InlineString("queen")

    v1 = model.input_embeddings[:, vocab.stoi[w1]]
    v2 = model.input_embeddings[:, vocab.stoi[w2]]
    sim = dot(v1, v2) / (norm(v1) * norm(v2))

    @test sim > 0.5
end


@testset "Training & Embeddings" begin
    text = "the king and the queen ruled the kingdom. the princess and the prince were loved by the people."
    model, vocab = Word2Vec_JL.Interface.train_word2vec(
        text; mode=:cbow, loss_type=:neg_sampling, dim=10, window_size=2, epochs=3, verbose=false
    )

    w1 = InlineString("king")
    w2 = InlineString("queen")

    v1 = model.input_embeddings[:, vocab.stoi[w1]]
    v2 = model.input_embeddings[:, vocab.stoi[w2]]
    sim = dot(v1, v2) / (norm(v1) * norm(v2))

    @test sim > 0.5  # Zakładamy, że podobieństwo jest przynajmniej umiarkowane
end

@testset "Preprocessing" begin
    raw = """
        Hello! This is <b>Pauli</b>'s test-email: pauli@openai.com.
        Visit: https://test.com/path?q=hello.    Thanks!!! 🤖🤯💥
    """

    expected = InlineString.([
        "hello", "this", "is", "pauli", "s", "test", "email", "visit", "thanks"
    ])

    result = preprocess_text(raw)

    @test result == expected
end

@testset "Training pipeline" begin
    txt = "the king and the queen ruled the kingdom"
    model, vocab = train_word2vec(txt; epochs=1, dim=10)
    @test size(model.input_embeddings, 1) == length(vocab.stoi)
end



