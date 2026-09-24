#include "llama-hot-experts.h"

#include "llama-impl.h"
#include "llama-model.h"

#include "ggml-backend.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>

llama_hot_experts::llama_hot_experts(int n_layer, int n_expert, const llama_hot_experts_config & cfg) :
    cfg(cfg), n_layer(n_layer), n_expert(n_expert),
    recent((size_t) n_layer * n_expert, 0.0f),
    total ((size_t) n_layer * n_expert, 0.0f) {
    if (!cfg.trace_path.empty()) {
        trace = fopen(cfg.trace_path.c_str(), "ab");
        if (!trace) {
            LLAMA_LOG_WARN("%s: cannot open trace file %s\n", __func__, cfg.trace_path.c_str());
        } else {
            LLAMA_LOG_INFO("%s: routing trace appended to %s\n", __func__, cfg.trace_path.c_str());
        }
    }
}

llama_hot_experts::~llama_hot_experts() {
    if (!cfg.stats_path.empty() && tokens_seen > 0) {
        flush();
    }
    if (trace) {
        fclose(trace);
    }
}

bool llama_hot_experts::load_stats() {
    if (cfg.stats_path.empty()) {
        return false;
    }
    std::ifstream f(cfg.stats_path);
    if (!f) {
        return false;
    }
    std::string line;
    size_t n_lines = 0;
    while (std::getline(f, line)) {
        if (line.empty()) {
            continue;
        }
        if (line[0] == '#') {
            int nl = 0, ne = 0;
            if (sscanf(line.c_str(), "# llama-hot-experts-stats n_layer=%d n_expert=%d", &nl, &ne) == 2) {
                if (nl != n_layer || ne != n_expert) {
                    LLAMA_LOG_WARN("%s: %s is for n_layer=%d n_expert=%d, this model has %d/%d: ignored\n",
                            __func__, cfg.stats_path.c_str(), nl, ne, n_layer, n_expert);
                    return false;
                }
            }
            continue;
        }
        // "cpu-prof-experts: <layer> <count> ..." (the same line format as the CPU profiler and expert-place.py)
        const char * p = line.c_str();
        const char * tag = "cpu-prof-experts:";
        if (strncmp(p, tag, strlen(tag)) == 0) {
            p += strlen(tag);
        }
        std::istringstream ss(p);
        int il = -1;
        if (!(ss >> il) || il < 0 || il >= n_layer) {
            continue;
        }
        // the slow histogram is the file; the fast one starts at a quarter of it so a new workload takes over
        // within a few thousand generated tokens while the shipped set is not churned out by the first few hundred
        float v = 0.0f;
        for (int e = 0; e < n_expert && (ss >> v); e++) {
            total[(size_t) il * n_expert + e]  = v;
            recent[(size_t) il * n_expert + e] = 0.25f * v;
        }
        n_lines++;
    }
    stats_loaded = n_lines > 0;
    if (stats_loaded) {
        double sum = 0.0;
        for (float v : total) {
            sum += v;
        }
        LLAMA_LOG_INFO("%s: %s: %zu layers, %.0f routed slots of history\n", __func__, cfg.stats_path.c_str(), n_lines, sum);
    }
    return stats_loaded;
}

std::map<int, std::vector<int32_t>> llama_hot_experts::placement(const std::map<int, size_t> & bytes_per_expert, size_t budget_bytes) const {
    struct cand {
        float  score;
        int    il;
        int    e;
        size_t bytes;
    };
    std::vector<cand> cands;
    cands.reserve(bytes_per_expert.size() * n_expert);
    for (const auto & [il, bytes] : bytes_per_expert) {
        if (il < 0 || il >= n_layer || bytes == 0) {
            continue;
        }
        for (int e = 0; e < n_expert; e++) {
            const float c = total[(size_t) il * n_expert + e];
            if (c > 0.0f) {
                cands.push_back({ c / (float) bytes, il, e, bytes });
            }
        }
    }
    std::sort(cands.begin(), cands.end(), [](const cand & a, const cand & b) { return a.score > b.score; });

    std::map<int, std::vector<int32_t>> spec;
    size_t used = 0;
    for (const auto & c : cands) {
        if (used + c.bytes > budget_bytes) {
            continue;
        }
        spec[c.il].push_back(c.e);
        used += c.bytes;
    }
    return spec;
}

double llama_hot_experts::coverage(const std::map<int, std::vector<int32_t>> & spec) const {
    double all = 0.0, hot = 0.0;
    for (int il = 0; il < n_layer; il++) {
        for (int e = 0; e < n_expert; e++) {
            all += total[(size_t) il * n_expert + e];
        }
    }
    for (const auto & [il, experts] : spec) {
        for (int32_t e : experts) {
            if (il >= 0 && il < n_layer && e >= 0 && e < n_expert) {
                hot += total[(size_t) il * n_expert + e];
            }
        }
    }
    return all > 0.0 ? hot / all : 0.0;
}

void llama_hot_experts::bind_layer(int il, llama_layer * layer, std::vector<int32_t> * slots) {
    layer_state st;
    st.layer = layer;
    st.slots = slots;
    st.hot_map.assign(n_expert, -1);
    st.cold_map.resize(n_expert);
    st.hot_mask.assign(n_expert, 0.0f);
    for (int e = 0; e < n_expert; e++) {
        st.cold_map[e] = e;
    }
    for (size_t s = 0; s < slots->size(); s++) {
        const int32_t e = (*slots)[s];
        if (e >= 0 && e < n_expert) {
            st.hot_map[e]  = (int32_t) s;
            st.cold_map[e] = -1;
            st.hot_mask[e] = 1.0f;
        }
    }
    st.protect_until.assign(slots->size(), 0);
    layers[il] = std::move(st);
}

void llama_hot_experts::begin_ubatch(uint32_t n_tokens) {
    std::lock_guard<std::mutex> lock(mtx);
    obs_weight = n_tokens >= cfg.prefill_min_tokens ? cfg.prefill_weight : 1.0f;
    if (trace) {
        const int32_t tag = -1;
        fwrite(&tag, sizeof(tag), 1, trace);
        fwrite(&n_tokens, sizeof(n_tokens), 1, trace);
    }
}

void llama_hot_experts::observe(int il, const int32_t * ids, size_t n) {
    if (il < 0 || il >= n_layer) {
        return;
    }
    std::lock_guard<std::mutex> lock(mtx);
    float * r = recent.data() + (size_t) il * n_expert;
    float * t = total.data()  + (size_t) il * n_expert;
    const auto it = layers.find(il);
    const int32_t * hot_map = it != layers.end() ? it->second.hot_map.data() : nullptr;
    const float w = obs_weight;
    for (size_t i = 0; i < n; i++) {
        const int32_t e = ids[i];
        if (e < 0 || e >= n_expert) {
            continue;
        }
        r[e] += w;
        t[e] += w;
        win_obs++;
        if (hot_map && hot_map[e] >= 0) {
            win_hit++;
        }
    }
    if (trace) {
        const int32_t  l  = il;
        const uint32_t nn = (uint32_t) n;
        trace_buf.resize(n);
        for (size_t i = 0; i < n; i++) {
            trace_buf[i] = (int16_t) ids[i];
        }
        fwrite(&l,  sizeof(l),  1, trace);
        fwrite(&nn, sizeof(nn), 1, trace);
        fwrite(trace_buf.data(), sizeof(int16_t), n, trace);
    }
}

void llama_hot_experts::decay(std::vector<float> & h, float factor) {
    for (float & v : h) {
        v *= factor;
    }
}

void llama_hot_experts::upload_maps(layer_state & st) {
    const llama_layer & layer = *st.layer;
    ggml_backend_tensor_set(layer.ffn_hot_map,  st.hot_map.data(),  0, n_expert*sizeof(int32_t));
    ggml_backend_tensor_set(layer.ffn_cold_map, st.cold_map.data(), 0, n_expert*sizeof(int32_t));
    ggml_backend_tensor_set(layer.ffn_hot_mask, st.hot_mask.data(), 0, n_expert*sizeof(float));
}

void llama_hot_experts::apply_swap(layer_state & st, int il, int slot, int expert, bool tensors_valid) {
    const llama_layer & layer = *st.layer;
    const int32_t victim = (*st.slots)[slot];

    if (victim >= 0 && victim < n_expert) {
        st.hot_map[victim]  = -1;
        st.cold_map[victim] = victim;
        st.hot_mask[victim] = 0.0f;
    }
    st.hot_map[expert]  = slot;
    st.cold_map[expert] = -1;
    st.hot_mask[expert] = 1.0f;
    (*st.slots)[slot] = expert;
    st.protect_until[slot] = tick_no + cfg.protect_ticks;

    if (tensors_valid) {
        auto copy = [&](ggml_tensor * src, ggml_tensor * dst) {
            const size_t bytes = src->nb[2];
            if (ggml_backend_buffer_is_host(src->buffer)) {
                ggml_backend_tensor_set(dst, (const char *) src->data + (size_t) expert*bytes, (size_t) slot*bytes, bytes);
            } else {
                tmp.resize(bytes);
                ggml_backend_tensor_get(src, tmp.data(), (size_t) expert*bytes, bytes);
                ggml_backend_tensor_set(dst, tmp.data(), (size_t) slot*bytes, bytes);
            }
            win_bytes += bytes;
        };
        copy(layer.ffn_gate_exps, layer.ffn_gate_hot);
        copy(layer.ffn_up_exps,   layer.ffn_up_hot);
        copy(layer.ffn_down_exps, layer.ffn_down_hot);
    }

    win_swaps++;
    swaps_total++;
    GGML_UNUSED(il);
}

size_t llama_hot_experts::n_bound() const {
    size_t n = 0;
    for (const auto & [il, st] : layers) {
        n += st.slots->size();
    }
    return n;
}

void llama_hot_experts::tick(uint32_t n_tokens, bool tensors_valid) {
    std::lock_guard<std::mutex> lock(mtx);

    const double weighted = (double) n_tokens * (n_tokens >= cfg.prefill_min_tokens ? cfg.prefill_weight : 1.0f);

    tokens_seen    += n_tokens;
    tokens_tick    += n_tokens;
    tokens_flush   += n_tokens;
    tokens_log     += n_tokens;
    tokens_decay_r += weighted;
    tokens_decay_t += weighted;

    // decay, applied in steps of a quarter half-life
    if (tokens_decay_r >= cfg.recent_half_life / 4) {
        decay(recent, std::pow(0.5f, (float) (tokens_decay_r / cfg.recent_half_life)));
        tokens_decay_r = 0;
    }
    if (tokens_decay_t >= cfg.total_half_life / 4) {
        decay(total, std::pow(0.5f, (float) (tokens_decay_t / cfg.total_half_life)));
        tokens_decay_t = 0;
    }

    const bool policy = cfg.swaps_enabled && !layers.empty() && (tokens_tick >= cfg.tick_tokens || n_tokens >= 32);
    if (policy) {
        tokens_tick = 0;
        tick_no++;

        std::vector<swap_candidate> cands;
        for (auto & [il, st] : layers) {
            const float * r = recent.data() + (size_t) il * n_expert;
            const size_t n_hot = st.slots->size();
            if (n_hot == 0) {
                continue;
            }
            // residents by ascending score, outsiders by descending score
            std::vector<int> res_slots(n_hot);
            for (size_t s = 0; s < n_hot; s++) {
                res_slots[s] = (int) s;
            }
            std::sort(res_slots.begin(), res_slots.end(), [&](int a, int b) {
                const int32_t ea = (*st.slots)[a], eb = (*st.slots)[b];
                const float sa = ea >= 0 ? r[ea] : -1.0f, sb = eb >= 0 ? r[eb] : -1.0f;
                return sa < sb;
            });
            std::vector<int> out;
            out.reserve(n_expert);
            for (int e = 0; e < n_expert; e++) {
                if (st.hot_map[e] < 0 && r[e] > 0.0f) {
                    out.push_back(e);
                }
            }
            std::partial_sort(out.begin(), out.begin() + std::min(out.size(), (size_t) cfg.max_swaps_per_layer), out.end(),
                    [&](int a, int b) { return r[a] > r[b]; });

            size_t vi = 0;
            for (size_t k = 0; k < out.size() && k < cfg.max_swaps_per_layer; k++) {
                // next evictable resident
                while (vi < n_hot && st.protect_until[res_slots[vi]] > tick_no) {
                    vi++;
                }
                if (vi >= n_hot) {
                    break;
                }
                const int     slot   = res_slots[vi];
                const int32_t victim = (*st.slots)[slot];
                const float   sv     = victim >= 0 ? r[victim] : 0.0f;
                const float   sc     = r[out[k]];
                if (sc <= cfg.hysteresis_ratio * sv + cfg.hysteresis_margin) {
                    break;
                }
                cands.push_back({ il, slot, out[k], sc - sv });
                vi++;
            }
        }

        std::sort(cands.begin(), cands.end(), [](const swap_candidate & a, const swap_candidate & b) { return a.gain > b.gain; });

        size_t n_done = 0;
        std::vector<int> touched;
        for (const auto & c : cands) {
            if (n_done >= cfg.max_swaps_per_tick) {
                break;
            }
            layer_state & st = layers[c.il];
            apply_swap(st, c.il, c.slot, c.expert, tensors_valid);
            if (std::find(touched.begin(), touched.end(), c.il) == touched.end()) {
                touched.push_back(c.il);
            }
            n_done++;
        }
        for (int il : touched) {
            upload_maps(layers[il]);
        }
    }

    if (cfg.log_tokens > 0 && tokens_log >= cfg.log_tokens) {
        LLAMA_LOG_INFO("hot experts: %" PRIu64 " tokens, coverage %.1f%% over the last %" PRIu64 " (%zu resident), %" PRIu64 " swaps (%.1f MiB), %" PRIu64 " total\n",
                tokens_seen, win_obs > 0 ? 100.0 * (double) win_hit / (double) win_obs : 0.0, tokens_log, n_bound(),
                win_swaps, win_bytes / 1024.0 / 1024.0, swaps_total);
        tokens_log = 0;
        win_obs = win_hit = win_swaps = win_bytes = 0;
    }

    if (!cfg.stats_path.empty() && tokens_flush >= cfg.flush_tokens) {
        tokens_flush = 0;
        flush();
    }
    if (trace && policy) {
        fflush(trace);
    }
}

void llama_hot_experts::flush() {
    if (cfg.stats_path.empty()) {
        return;
    }
    const std::string tmp_path = cfg.stats_path + ".tmp";
    {
        std::ofstream f(tmp_path, std::ios::trunc);
        if (!f) {
            LLAMA_LOG_WARN("%s: cannot write %s\n", __func__, tmp_path.c_str());
            return;
        }
        f << "# llama-hot-experts-stats n_layer=" << n_layer << " n_expert=" << n_expert << " tokens=" << tokens_seen << "\n";
        f << "# one line per layer, decayed routed-slot counts per expert (half-life " << (uint64_t) cfg.total_half_life
          << " tokens); readable by customware/expert-place.py\n";
        for (int il = 0; il < n_layer; il++) {
            const float * t = total.data() + (size_t) il * n_expert;
            f << "cpu-prof-experts: " << il;
            for (int e = 0; e < n_expert; e++) {
                f << ' ' << (uint64_t) std::llround(t[e]);
            }
            f << "\n";
        }
    }
    std::remove(cfg.stats_path.c_str());
    if (std::rename(tmp_path.c_str(), cfg.stats_path.c_str()) != 0) {
        LLAMA_LOG_WARN("%s: cannot rename %s to %s\n", __func__, tmp_path.c_str(), cfg.stats_path.c_str());
    }
}
