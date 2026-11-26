#ifndef PLAMN_CORE_GEOMETRY_INDEXER_HPP
#define PLAMN_CORE_GEOMETRY_INDEXER_HPP

#include "damn/core/vector.hpp"
#include "damn/core/memory/span.hpp"
#include "damn/core/def/damn_config.hpp"

#include "plamn/core/def.hpp"
#include "plamn/core/data/particles/particles_def.hpp"

#include <vector>
#include <cassert>
#include <cstddef>
#include <algorithm>
#include <initializer_list>
#include <iterator>

namespace plamn::core
{

struct IndexerOptions
{
    using value_type         = int;
    damn::AllocatorMode alloc_mode = damn::AllocatorMode::CPU;
    StorageMode storage_mode = StorageMode::VECTOR;

    static constexpr IndexerOptions FROM(auto const& opts)
    {
        return {opts.alloc_mode, opts.storage_mode};
    }
};


// Forward declaration so IndexerStorage<VECTOR> can name Indexer<SPAN> as its base.
template<auto opts = IndexerOptions{}>
class Indexer;
template<auto storage_mode, auto opts>
struct IndexerStorage;


// SPAN — zero-allocation view, used in GPU kernels

template<auto opts>
struct IndexerStorage<StorageMode::SPAN, opts>
{
    IndexerStorage()                                     = default;
    IndexerStorage(IndexerStorage const&)                = default;
    IndexerStorage(IndexerStorage&&)                     = default;
    IndexerStorage& operator=(IndexerStorage const&)     = default;
    IndexerStorage& operator=(IndexerStorage&&) noexcept = default;

    MKN_KUL_FN_SIG_RET auto data() const DAMN_ALL { return indexes_.data(); }
    MKN_KUL_FN_SIG_RET auto data() DAMN_ALL { return indexes_.data(); }
    MKN_KUL_FN_SIG_RET std::size_t size() const DAMN_ALL { return indexes_.size(); }
    MKN_KUL_FN_SIG_RET bool is_empty() const DAMN_ALL { return indexes_.size() == 0; }

    MKN_KUL_FN_SIG_RET auto begin() DAMN_ALL { return indexes_.begin(); }
    MKN_KUL_FN_SIG_RET auto begin() const DAMN_ALL { return indexes_.begin(); }
    MKN_KUL_FN_SIG_RET auto cbegin() const DAMN_ALL { return indexes_.begin(); }
    MKN_KUL_FN_SIG_RET auto end() DAMN_ALL { return indexes_.end(); }
    MKN_KUL_FN_SIG_RET auto end() const DAMN_ALL { return indexes_.end(); }
    MKN_KUL_FN_SIG_RET auto cend() const DAMN_ALL { return indexes_.end(); }

    MKN_KUL_FN_SIG_RET auto& operator[](std::size_t i) DAMN_ALL { return indexes_[i]; }
    MKN_KUL_FN_SIG_RET auto& operator[](std::size_t i) const DAMN_ALL { return indexes_[i]; }
    MKN_KUL_FN_SIG_RET auto& back() DAMN_ALL { return indexes_[indexes_.size() - 1]; }
    MKN_KUL_FN_SIG_RET auto& front() const DAMN_ALL { return indexes_[0]; }

    void sort() { std::sort(indexes_.begin(), indexes_.end()); }
    void swap(auto const& a, auto const& b) { std::swap(indexes_[a], indexes_[b]); }

    MKN_KUL_FN_SIG_RET bool is_indexed(std::size_t itemIndex) const
    {
        return std::end(indexes_) != std::find(std::begin(indexes_), std::end(indexes_), itemIndex);
    }

    void updateIndex(std::size_t oldIndex, std::size_t newIndex)
    {
        auto it = std::find(std::begin(indexes_), std::end(indexes_), oldIndex);
        if (it != std::end(indexes_))
            *it = newIndex;
    }

    void set_from(auto& indexer) { indexes_ = {indexer.data(), indexer.size()}; }
    void clear() { /* noop — view does not own */ }

protected:
    damn::core::Span<std::size_t> indexes_;
};


// VECTOR — CPU-owning storage. Inherits Indexer<SPAN> so static_cast<Indexer<SPAN>*> is valid
// for GPU dispatch. All CPU operations work directly on vec_; call sync() before GPU use to
// populate the inherited span.

template<auto opts>
struct IndexerStorage<StorageMode::VECTOR, opts>
    : Indexer<IndexerOptions{opts.alloc_mode, StorageMode::SPAN}>
{
    using View       = Indexer<IndexerOptions{opts.alloc_mode, StorageMode::SPAN}>;
    using vec_helper = damn::Vector<std::size_t, opts.alloc_mode>;
    using vec_t      = typename vec_helper::vector_t;

    IndexerStorage() = default;

    IndexerStorage(IndexerStorage const& other)
        : View{}
        , vec_{vec_helper::from(other.vec_)}
    {
    }

    IndexerStorage(IndexerStorage&& other) noexcept
        : View{}
        , vec_{vec_helper::from(std::move(other.vec_))}
    {
    }

    IndexerStorage& operator=(IndexerStorage const& other)
    {
        vec_helper::copy(vec_, other.vec_);
        return *this;
    }

    IndexerStorage& operator=(IndexerStorage&& other) noexcept
    {
        vec_ = vec_helper::from(std::move(other.vec_));
        return *this;
    }

    // Populate the inherited SPAN view — call before passing to a GPU kernel.
    void sync() { View::indexes_ = {vec_.data(), vec_.size()}; }


    MKN_KUL_FN_SIG_RET auto data() const { return vec_.data(); }
    MKN_KUL_FN_SIG_RET auto data() { return vec_.data(); }
    MKN_KUL_FN_SIG_RET std::size_t size() const { return vec_.size(); }
    MKN_KUL_FN_SIG_RET bool is_empty() const { return vec_.empty(); }

    MKN_KUL_FN_SIG_RET auto begin() { return vec_.begin(); }
    MKN_KUL_FN_SIG_RET auto begin() const { return vec_.begin(); }
    MKN_KUL_FN_SIG_RET auto cbegin() const { return vec_.cbegin(); }
    MKN_KUL_FN_SIG_RET auto end() { return vec_.end(); }
    MKN_KUL_FN_SIG_RET auto end() const { return vec_.end(); }
    MKN_KUL_FN_SIG_RET auto cend() const { return vec_.cend(); }

    MKN_KUL_FN_SIG_RET auto& operator[](std::size_t i) { return vec_[i]; }
    MKN_KUL_FN_SIG_RET auto& operator[](std::size_t i) const { return vec_[i]; }
    MKN_KUL_FN_SIG_RET auto& back() { return vec_.back(); }
    MKN_KUL_FN_SIG_RET auto& front() const { return vec_.front(); }

    void sort() { std::sort(vec_.begin(), vec_.end()); }
    void swap(auto const& a, auto const& b) { std::swap(vec_[a], vec_[b]); }

    MKN_KUL_FN_SIG_RET bool is_indexed(std::size_t itemIndex) const
    {
        return std::end(vec_) != std::find(std::begin(vec_), std::end(vec_), itemIndex);
    }

    void updateIndex(std::size_t oldIndex, std::size_t newIndex)
    {
        auto it = std::find(vec_.begin(), vec_.end(), oldIndex);
        if (it != vec_.end())
            *it = newIndex;
    }

    void set_from(auto& indexer) { vec_.assign(indexer.data(), indexer.data() + indexer.size()); }

    void add(std::size_t itemIndex) { vec_.push_back(itemIndex); }

    void remove(std::size_t itemIndex)
    {
        auto it = std::find(vec_.begin(), vec_.end(), itemIndex);
        if (it != vec_.end())
            vec_.erase(it);
        assert(!is_indexed(itemIndex));
    }

    void empty() { vec_.resize(0); }
    void clear() { vec_.clear(); }
    void resize(std::size_t s) { vec_.resize(s); }
    MKN_KUL_FN_SIG_RET std::size_t capacity() const { return vec_.capacity(); }

    vec_t vec_;
};



template<auto opts>
class Indexer : public IndexerStorage<opts.storage_mode, opts>
{
public:
    using IndexerStorage<opts.storage_mode, opts>::IndexerStorage;
};


} // namespace plamn::core


#endif
