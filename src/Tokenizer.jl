module Tokenizer

using InlineStrings

export clean_and_tokenize

"""
    clean_and_tokenize(text::String; group_by_sentences::Bool=false) 
        -> Union{Vector{InlineString}, Vector{Vector{InlineString}}}

Cleans and tokenizes the input text.

# Arguments
- `text::String`: Raw text input.
- `group_by_sentences::Bool=false`: 
    - If `false` (default): returns a flat list of tokens (`Vector{InlineString}`).
    - If `true`: returns a list of tokenized sentences (`Vector{Vector{InlineString}}`).

# Returns
- `Vector{InlineString}` if `group_by_sentences=false`.
- `Vector{Vector{InlineString}}` if `group_by_sentences=true`.

# Cleaning includes:
- Lowercasing text,
- Removing emails, URLs, HTML/XML tags,
- Removing standalone numbers,
- Removing special characters and punctuation (except hyphens and apostrophes),
- Removing extra whitespace.
"""
function clean_and_tokenize(text::String; group_by_sentences::Bool=false)
    # Convert to lowercase
    text = lowercase(text)

    # Initial split into sentences using punctuation delimiters
    raw_sentences = split(text, r"[\.!\?]+")

    if !group_by_sentences
        # Clean and tokenize entire text as a single sequence
        text = replace(text, r"\b[\w\.\-]+@[\w\.\-]+\.\w{2,}\b" => " ")
        text = replace(text, r"https?:\/\/\S+" => " ")
        text = replace(text, r"<[^>]+>" => " ")
        text = replace(text, r"&\w+;" => " ")
        text = replace(text, r"\b\d+\b" => " ")
        text = replace(text, r"[^a-z\s'-]" => " ")
        text = replace(text, r"\s+" => " ")
        text = strip(text)

        tokens = split(text)
        return InlineString.(tokens)

    else
        # Clean and tokenize each sentence separately
        processed_sentences = Vector{Vector{InlineString}}()

        for sentence in raw_sentences
            sentence = strip(sentence)
            if !isempty(sentence)
                sentence = replace(sentence, r"\b[\w\.\-]+@[\w\.\-]+\.\w{2,}\b" => " ")
                sentence = replace(sentence, r"https?:\/\/\S+" => " ")
                sentence = replace(sentence, r"<[^>]+>" => " ")
                sentence = replace(sentence, r"&\w+;" => " ")
                sentence = replace(sentence, r"\b\d+\b" => " ")
                sentence = replace(sentence, r"[^a-z\s'-]" => " ")
                sentence = replace(sentence, r"\s+" => " ")
                sentence = strip(sentence)

                tokens = split(sentence)
                if !isempty(tokens)
                    push!(processed_sentences, InlineString.(tokens))
                end
            end
        end

        return processed_sentences
    end
end

end # module



