module Huffman

using InlineStrings

export HuffmanNode, build_huffman_tree, generate_codes

"""
    struct HuffmanNode

A node in the Huffman tree.
Contains the word (if it's a leaf), its frequency, optional left/right children,
and an assigned unique index for training.
"""
mutable struct HuffmanNode
    token::InlineString
    freq::Int
    left::Union{Nothing, HuffmanNode}
    right::Union{Nothing, HuffmanNode}
    index::Union{Nothing, Int}  
end

"""
    build_huffman_tree(freqs::Dict{InlineString, Int}) -> HuffmanNode

Builds a Huffman tree from a dictionary of word frequencies.
Returns the root node.
"""
function build_huffman_tree(freqs::Dict{InlineString, Int})
    nodes = [HuffmanNode(w, f, nothing, nothing, nothing) for (w, f) in freqs]

    while length(nodes) > 1
        sort!(nodes, by=n -> n.freq)
        left, right = nodes[1], nodes[2]

        new = HuffmanNode(InlineString(""), left.freq + right.freq, left, right, nothing)
        push!(nodes, new)
        splice!(nodes, 1:2)
    end

    return nodes[1]
end

"""
    generate_codes(root::HuffmanNode) -> (Dicts with codes, paths, indices)

Generates:
- `codes`: binary Huffman codes for each word
- `paths`: list of output layer indices on the path to each word
- `index_map`: mapping from node (pointer) to assigned index
"""
function generate_codes(root::HuffmanNode)
    codes = Dict{InlineString, Vector{Int}}()
    paths = Dict{InlineString, Vector{Int}}()
    index_counter = Ref(1)

    node_to_index = IdDict{HuffmanNode, Int}()

    function traverse(n::HuffmanNode, code::Vector{Int}, path::Vector{Int})
        if isnothing(n.left) && isnothing(n.right)
            codes[n.token] = copy(code)
            paths[n.token] = copy(path)
            return
        end

        # Assign index to this internal node
        idx = index_counter[]
        n.index = idx
        node_to_index[n] = idx
        index_counter[] += 1 

        if n.left !== nothing
            traverse(n.left, push!(copy(code), 0), push!(copy(path), idx))
        end
        if n.right !== nothing
            traverse(n.right, push!(copy(code), 1), push!(copy(path), idx))
        end
    end

    traverse(root, Int[], Int[])
    return codes, paths
end


end # module




