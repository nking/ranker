#pragma once
#include <vector>
#include <memory>
#include "scann/scann_ops/scann_interface.h"

class ScannSearcher {
public:
    // Pass in the data (flattened vector) and dimensions
    ScannSearcher(const std::vector<float>& dataset, int dim);
    
    // Search returns vector of {id, distance}
    std::vector<std::pair<int, float>> Search(const std::vector<float>& query, int k);

    // Best practice: Serialize to disk for fast reload
    void SaveIndex(const std::string& path);
    void LoadIndex(const std::string& path);

private:
    std::unique_ptr<scann::ScannInterface> index_;
    int dim_;
};