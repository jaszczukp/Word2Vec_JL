module Word2Vec_JL

using InlineStrings
using Serialization

include("Huffman.jl")
include("Tokenizer.jl")
include("Vocabulary.jl")
include("DataLoader.jl")
include("Word2VecModel.jl")
include("Trainer.jl")
include("Evaluation.jl")
include("Interface.jl")

using .Tokenizer
using .Vocabulary
using .DataLoader
using .Word2VecModel
using .Trainer
using .Huffman
using .Evaluation
using .Interface

# Public API – high-level only 
export Word2Vec
export save_model, load_model, train!, predict
export cosine_similarity, get_vector, most_similar, analogy

# Considered internal / expert API 
export Vocab, build_vocab
export clean_and_tokenize

end # module



