
// Apache-2.0
#include <unordered_map>
#include <vector>
#include <cstdint>
#include <algorithm>

// Map raw labels (component ids) to [0..K-1] compact ids, filter cluster sizes
// Returns vector of cluster_id for each point, or -1 for noise/filtered
std::vector<int32_t> compress_and_filter_labels(
  const std::vector<int32_t>& raw, int min_sz, int max_sz)
{
  std::unordered_map<int32_t, int> counts;
  counts.reserve(raw.size()/4+1);
  for (auto v : raw) if (v >= 0) ++counts[v];
  // build mapping for those within [min,max]
  std::unordered_map<int32_t, int32_t> remap;
  remap.reserve(counts.size());
  int32_t cur = 0;
  for (auto& kv : counts) {
    if (kv.second >= min_sz && kv.second <= max_sz) {
      remap[kv.first] = cur++;
    }
  }
  std::vector<int32_t> out(raw.size(), -1);
  for (size_t i=0;i<raw.size();++i) {
    auto it = remap.find(raw[i]);
    if (it != remap.end()) out[i] = it->second;
  }
  return out;
}
