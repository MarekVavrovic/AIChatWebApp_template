using Microsoft.Extensions.AI;
using Microsoft.Extensions.VectorData;

namespace BlazorNETaiNewTemplate.Services;

public class SemanticSearch(
    IEmbeddingGenerator<string, Embedding<float>> embeddingGenerator,
    IVectorStore vectorStore)
{
    // The SemanticSearch service uses the embeddings model to convert the searchPhrase (text) into a vector
    // and then queries the vector store for the nearest matching records.
    // If a filenameFilter is provided, it filters the results to only include records from that filename.
    public async Task<IReadOnlyList<SemanticSearchRecord>> SearchAsync(string text, string? filenameFilter, int maxResults)
    {
        var queryEmbedding = await embeddingGenerator.GenerateEmbeddingVectorAsync(text);
        var vectorCollection = vectorStore.GetCollection<string, SemanticSearchRecord>("data-blazornetainewtemplate-ingested");
        var filter = filenameFilter is { Length: > 0 }
            ? new VectorSearchFilter().EqualTo(nameof(SemanticSearchRecord.FileName), filenameFilter)
            : null;

        var nearest = await vectorCollection.VectorizedSearchAsync(queryEmbedding, new VectorSearchOptions
        {
            Top = maxResults,
            Filter = filter,
        });
        var results = new List<SemanticSearchRecord>();
        await foreach (var item in nearest.Results)
        {
            results.Add(item.Record);
        }

        return results;
    }
}
