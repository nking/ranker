#include "searcher.h"
#include "scann/scann_ops/scann_builder.h"

ScannSearcher::ScannSearcher(const std::vector<float>& dataset, int dim) : dim_(dim) {
    // 1. Configure the Builder
    // We use "dot_product" as the distance measure
    auto builder = scann::ScannBuilder(dataset, 10, "dot_product");

    // 2. Add Partitioning (Tree) for speed
    // 100 leaves is a good starting point for moderate datasets
    builder.set_tree(100, scann::PartitioningType::SPHERICAL, 
                     scann::TrainingThreads(4), scann::DistanceMeasure::DotProduct);

    // 3. Add Asymmetric Hashing (AH) for memory efficiency
    // These parameters (16, 8) are standard defaults for quality/speed trade-off
    builder.set_asymmetric_hash(16, 8);

    // 4. Build the interface
    index_ = builder.Build();
}

std::vector<std::pair<int, float>> ScannSearcher::Search(const std::vector<float>& query, int k) {
    std::vector<int32_t> neighbor_ids;
    std::vector<float> distances;

    // Perform the search
    index_->Search(query, k, &neighbor_ids, &distances);

    std::vector<std::pair<int, float>> results;
    for (size_t i = 0; i < neighbor_ids.size(); ++i) {
        results.push_back({neighbor_ids[i], distances[i]});
    }
    return results;
}