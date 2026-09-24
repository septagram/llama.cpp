#pragma once

// customware: adaptive hot experts (entry 004).
//
// Collects the router's choices for every token (llama_context reads the per-layer top-k ids back after each ubatch),
// keeps two decayed histograms per layer (a fast one that drives runtime swaps, a slow one that is persisted and
// drives the placement at the next start), and swaps experts in and out of a layer's GPU slots between graph
// computations. The slot -> expert lists are the model's placement spec, so a refill of parked tensors (entry 005)
// always copies the current set.

#include "ggml.h"

#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <vector>

struct llama_layer;

struct llama_hot_experts_config {
    std::string stats_path;                 // LLAMA_HOT_EXPERTS_STATS: histogram file, read at start, rewritten periodically
    size_t      budget_bytes        = 0;    // LLAMA_HOT_EXPERTS_MIB: placement budget from the stats (0 = use the placement file)
    bool        swaps_enabled       = true; // LLAMA_HOT_EXPERTS_SWAP=0 turns the runtime swaps off (collect only)
    uint32_t    tick_tokens         = 64;   // swap policy runs every this many decoded tokens (and after every prefill ubatch)
    uint32_t    max_swaps_per_tick  = 48;   // global budget per tick (~3 MB each)
    uint32_t    max_swaps_per_layer = 4;    // per layer per tick
    uint32_t    protect_ticks       = 2;    // a freshly swapped-in expert cannot be evicted for this many ticks
    float       hysteresis_ratio    = 1.5f; // swap when recent[cand] > ratio * recent[victim] + margin
    float       hysteresis_margin   = 2.0f;
    float       recent_half_life    = 4096.0f;   // weighted tokens
    float       total_half_life     = 262144.0f; // weighted tokens
    uint32_t    prefill_min_tokens  = 32;        // ubatches of at least this many tokens count as prefill
    float       prefill_weight      = 1.0f/16;   // weight of a prefill token in the histograms and the decay clock
    uint32_t    flush_tokens        = 8192; // rewrite the stats file every this many tokens
    uint32_t    log_tokens          = 4096; // log coverage / swaps every this many tokens (0 = never)
    std::string trace_path;                 // LLAMA_HOT_EXPERTS_TRACE: per-token routing ids, binary, for offline replay
};

class llama_hot_experts {
public:
    llama_hot_experts(int n_layer, int n_expert, const llama_hot_experts_config & cfg);
    ~llama_hot_experts();

    // read the stats file; returns false if it is missing or does not match the model
    bool load_stats();

    // greedy byte-aware placement from the slow histogram: for each eligible layer its expert byte size,
    // returns layer -> experts (sorted by score) that fit the budget
    std::map<int, std::vector<int32_t>> placement(const std::map<int, size_t> & bytes_per_expert, size_t budget_bytes) const;

    // predicted share of routed slots the given placement serves on the slow histogram
    double coverage(const std::map<int, std::vector<int32_t>> & spec) const;

    // bind a layer's runtime state: the hot tensors, the full tensors, and the (mutable) slot -> expert list of the spec
    void bind_layer(int il, llama_layer * layer, std::vector<int32_t> * slots);

    // routing observations of one layer for one ubatch: ids[n_expert_used * n_tokens]. Big ubatches (prefill) are
    // weighted down: the hot set pays off per decoded token, and a 22k prefill must not wipe the decode history
    void begin_ubatch(uint32_t n_tokens);
    void observe(int il, const int32_t * ids, size_t n);

    // once per ubatch after all layers were observed. tensors_valid = false means the hot tensors' storage is
    // currently borrowed (entry 005): swaps then update the spec and the maps only; the next refill copies the data
    void tick(uint32_t n_tokens, bool tensors_valid);

    void flush();

    bool has_stats() const { return stats_loaded; }
    const std::string & stats_path() const { return cfg.stats_path; }

private:
    struct layer_state {
        llama_layer *          layer = nullptr;
        std::vector<int32_t> * slots = nullptr;   // slot -> expert (the spec)
        std::vector<int32_t>   hot_map;           // expert -> slot or -1
        std::vector<int32_t>   cold_map;          // expert or -1
        std::vector<float>     hot_mask;
        std::vector<uint32_t>  protect_until;     // per slot, tick number
    };

    struct swap_candidate {
        int   il;
        int   slot;
        int   expert;
        float gain;
    };

    void   decay(std::vector<float> & h, float factor);
    void   apply_swap(layer_state & st, int il, int slot, int expert, bool tensors_valid);
    void   upload_maps(layer_state & st);
    size_t n_bound() const;

    llama_hot_experts_config cfg;

    int n_layer  = 0;
    int n_expert = 0;

    std::vector<float> recent; // [n_layer * n_expert]
    std::vector<float> total;  // [n_layer * n_expert]
    std::map<int, layer_state> layers;

    bool     stats_loaded   = false;
    float    obs_weight     = 1.0f; // weight of the current ubatch's observations
    uint64_t tokens_seen    = 0;   // observed tokens since start (raw)
    uint64_t tokens_tick    = 0;   // since the last policy tick (raw)
    uint64_t tokens_flush   = 0;
    uint64_t tokens_log     = 0;
    double   tokens_decay_r = 0;   // weighted
    double   tokens_decay_t = 0;   // weighted
    uint32_t tick_no        = 0;

    // window statistics for the log line
    uint64_t win_obs   = 0;
    uint64_t win_hit   = 0;
    uint64_t win_swaps = 0;
    uint64_t win_bytes = 0;
    uint64_t swaps_total = 0;

    std::vector<uint8_t> tmp;
    std::mutex mtx;

    // trace file: records of {int32 -1, uint32 n_tokens} per ubatch, then {int32 layer, uint32 n, int16 ids[n]} per layer
    FILE * trace = nullptr;
    std::vector<int16_t> trace_buf;
};
