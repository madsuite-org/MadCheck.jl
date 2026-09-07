

"""
    dulmage_mendelsohn(A, B, E)

Compute the Dulmage-Mendelsohn decomposition [DulmageandMendelsohn-1958](@cite) to a bipartite graph. 
A and B are expected to be the 2 sets of vertices (even if they both posess a certain element, the function will consider them different). E is the sets of edges.

Return
A named tuple with 2 fields A and B each a named tuple corresponding to the decomposition of A and B respectively,  each have the following fields:
-`oc`: Overconstrained elements of the set
-`uc`: Underconstrained elements of the set
-`sq`: Well defined elements of the set (square)

Reference:
[DulmageandMendelsohn-1958] Dulage and Mendelsohn - 1958 - Coverings of Bipartite Graphs
"""
function dulmage_mendelsohn(A::Vector{T}, B::Vector{T}, E::Vector{Tuple{T, T}}) where T
    n_a = length(A)
    n_b = length(B)

    # We convert the sets to int ids

    A_vertex_to_id = Dict(a => i for (i, a) in enumerate(A))
    B_vertex_to_id = Dict(b => i + n_a for (i, b) in enumerate(B))
    E_id = [(A_vertex_to_id[a], B_vertex_to_id[b]) for (a, b) in E]

    id_to_vertex = merge(Dict(i => a for (i, a) in enumerate(A)), Dict(i + n_a => b for (i, b) in enumerate(B)))

    # BitMatrix of the graph

    g_matrix = falses(n_a, n_b)
    for (a,b) in E_id
        g_matrix[a, b - n_a] = true
    end
    
    # max matching using BipartiteMatching
    matching, matched = BipartiteMatching.findmaxcardinalitybipartitematching(g_matrix)

    # Digraph creation
    digraph = Graphs.SimpleDiGraph(n_a + n_b)
    for (a, b) in E_id
        if matched[b - n_a] && haskey(matching, a) && matching[a] == b - n_a
            Graphs.add_edge!(digraph, (b, a))
        else
            Graphs.add_edge!(digraph, (a, b))            
        end
    end

    # Build under and over constrained sets
    A_id_S = Set(1:n_a)
    B_id_S = Set(n_a+1:n_a + n_b)

    digraphReverse = Graphs.reverse(digraph)
    under_con = Set{Int}()
    over_con = Set{Int}()

    unmatched_A = filter(x-> !(haskey(matching, x) && matched[matching[x]]), A_id_S)
    unmatched_B = filter(x-> !(matched[x - n_a]), B_id_S)

    for a in unmatched_A
        bfs_a = Set{Int64}(collect(BFSIterator(digraph,a)))
        under_con = union(under_con, Set(bfs_a))
    end
    for b in unmatched_B
        Rbfs_b = Set{Int64}(collect(BFSIterator(digraphReverse, b)))
        over_con = union(over_con, Set(Rbfs_b))
    end

    # Isolate the different sets
    oc_A = intersect(over_con, A_id_S)
    uc_A = intersect(under_con, A_id_S)
    sq_A = setdiff(A_id_S, union(oc_A, uc_A))

    oc_B = intersect(over_con, B_id_S)
    uc_B = intersect(under_con, B_id_S)
    sq_B = setdiff(B_id_S, union(oc_B, uc_B))

    return (A = (oc = [id_to_vertex[a] for a in oc_A], 
            uc = [id_to_vertex[a] for a in uc_A],
            sq = [id_to_vertex[a] for a in sq_A]), 
            B =  (oc = [id_to_vertex[b] for b in oc_B],
            uc = [id_to_vertex[b] for b in uc_B],
            sq = [id_to_vertex[b] for b in sq_B])
            )
end