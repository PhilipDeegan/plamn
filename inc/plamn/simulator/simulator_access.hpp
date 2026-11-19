#ifndef PLAMN_SIMULATOR_SIMULATOR_ACCESS_HPP
#define PLAMN_SIMULATOR_SIMULATOR_ACCESS_HPP

// plamn::Simulator: what tests and diagnostics read: domain values, particle counts, summary
// IWYU pragma: private, include "plamn/simulator/simulator.hpp"

#include "damn/amr/box_data.hpp"
#include "damn/amr/level_data.hpp"
#include "damn/core/geometry/box.hpp"
#include "damn/core/geometry/box_spans.hpp"
#include "damn/core/geometry/point.hpp"
#include "damn/mpi/scope_timing.hpp"

#include "plamn/simulator/simulator.hpp" // IWYU pragma: keep (the class)

#include "mpi.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <filesystem>
#include <format>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <type_traits>
#include <utility>
#include <vector>
// plamn::Simulator: what tests and diagnostics read: domain values, particle counts, summary
// IWYU pragma: private, include "plamn/simulator/simulator.hpp"

namespace plamn
{
template<auto opts>
std::vector<double> Simulator<opts>::domain_values(std::string const& name, std::size_t const ln)
{
    auto& ld = data_(ln);
    std::vector<double> values;
    for (std::size_t i = 0; i < damn::amr::size(ld); ++i)
    {
        auto const& cells = ld.patches[i].box;
        bool const found  = damn::amr::with_resource(ld.data[i], name, [&](auto& r) {
            if constexpr (damn::amr::BoxData<std::decay_t<decltype(r)>, dimension>)
                damn::core::for_each_point(domain_box(r, cells), [&](auto const& p) {
                    values.push_back(value(r, cells, p));
                });
        });
        if (!found)
            throw std::runtime_error("plamn Simulator: no resource " + name);
    }
    return values;
}

template<auto opts>
std::vector<double> Simulator<opts>::all_domain_values(std::string const& name,
                                                       std::size_t const ln)
{
    auto const mine = domain_values(name, ln);
    if (n_ranks_ == 1)
        return mine;

    auto& ld = data_(ln);
    // every rank's values and per patch sizes (a rank may have no patch)
    std::vector<int> sizes; // of my patches, in my level order
    for (std::size_t i = 0; i < damn::amr::size(ld); ++i)
        damn::amr::with_resource(ld.data[i], name, [&](auto& r) {
            if constexpr (damn::amr::BoxData<std::decay_t<decltype(r)>, dimension>)
                sizes.push_back(static_cast<int>(domain_box(r, ld.patches[i].box).size()));
        });

    auto const gather = [&]<typename T>(std::vector<T> const& v, MPI_Datatype const type) {
        int const n = static_cast<int>(v.size());
        std::vector<int> counts(n_ranks_), displs(n_ranks_);
        MPI_Allgather(&n, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
        std::partial_sum(counts.begin(), counts.end() - 1, displs.begin() + 1);
        std::vector<T> all(static_cast<std::size_t>(displs.back() + counts.back()));
        MPI_Allgatherv(v.data(), n, type, all.data(), counts.data(), displs.data(), type,
                       MPI_COMM_WORLD);
        return std::make_pair(all, displs);
    };
    auto const [values, value_at]   = gather(mine, MPI_DOUBLE);
    auto const [all_sizes, size_at] = gather(sizes, MPI_INT);

    // a rank's patches come in level order: walk the level, taking each patch's values from its
    //  rank's next ones
    std::vector<std::size_t> next_value(n_ranks_), next_size(n_ranks_);
    for (int r = 0; r < n_ranks_; ++r)
    {
        next_value[r] = static_cast<std::size_t>(value_at[r]);
        next_size[r]  = static_cast<std::size_t>(size_at[r]);
    }
    std::vector<double> ordered;
    ordered.reserve(values.size());
    for (auto const& patch : hierarchy_.levels[ln].patches)
    {
        auto const size = static_cast<std::size_t>(all_sizes[next_size[patch.rank]++]);
        auto const from = values.begin() + static_cast<std::ptrdiff_t>(next_value[patch.rank]);
        ordered.insert(ordered.end(), from, from + static_cast<std::ptrdiff_t>(size));
        next_value[patch.rank] += size;
    }
    return ordered;
}

template<auto opts>
std::vector<double> Simulator<opts>::level_values(std::string const& name, std::size_t const ln)
{
    auto& ld = data_(ln);
    std::vector<int> points; // dimension per point
    std::vector<double> values;
    for (std::size_t i = 0; i < damn::amr::size(ld); ++i)
    {
        auto const& cells = ld.patches[i].box;
        bool const found  = damn::amr::with_resource(ld.data[i], name, [&](auto& r) {
            if constexpr (damn::amr::BoxData<std::decay_t<decltype(r)>, dimension>)
                damn::core::for_each_point(domain_box(r, cells), [&](auto const& p) {
                    for (std::size_t d = 0; d < dimension; ++d)
                        points.push_back(p[d]);
                    values.push_back(value(r, cells, p));
                });
        });
        if (!found)
            throw std::runtime_error("plamn Simulator: no resource " + name);
    }
    if (n_ranks_ > 1)
    {
        auto const gather = [&]<typename T>(std::vector<T> const& v, MPI_Datatype const type) {
            int const n = static_cast<int>(v.size());
            std::vector<int> counts(n_ranks_), displs(n_ranks_);
            MPI_Allgather(&n, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
            std::partial_sum(counts.begin(), counts.end() - 1, displs.begin() + 1);
            std::vector<T> all(static_cast<std::size_t>(displs.back() + counts.back()));
            MPI_Allgatherv(v.data(), n, type, all.data(), counts.data(), displs.data(), type,
                           MPI_COMM_WORLD);
            return all;
        };
        points = gather(points, MPI_INT);
        values = gather(values, MPI_DOUBLE);
    }

    std::map<std::array<int, dimension>, double> by_point;
    for (std::size_t k = 0; k < values.size(); ++k)
    {
        std::array<int, dimension> p{};
        for (std::size_t d = 0; d < dimension; ++d)
            p[d] = points[k * dimension + d];
        by_point.emplace(p, values[k]); // the first patch's
    }
    std::vector<double> out;
    out.reserve(by_point.size());
    for (auto const& [p, v] : by_point)
        out.push_back(v);
    return out;
}

template<auto opts>
Simulator<opts>::~Simulator()
{
    if (!timing_)
        return;
    int mpi_on = 0, mpi_done = 0;
    MPI_Initialized(&mpi_on);
    MPI_Finalized(&mpi_done);
    if (!mpi_on or mpi_done)
        return;         // too late to write
    std::error_code ec; // pyphare makes it on rank 0, maybe not yet
    std::filesystem::create_directories(std::filesystem::path{timings_prefix}.parent_path(), ec);
    damn::mpi::stop_scope_timing();
}

template<auto opts>
std::string Simulator<opts>::summary()
{
    using Patch_t     = solver::HybridModel<opts>::Patch_t;
    auto constexpr gw = Patch_t::field_ghost_width;

    std::stringstream ss;
    unsigned long long total = 0;
    for (std::size_t ln = 0; ln < levels_.size(); ++ln)
    {
        auto& l = levels_[ln];
        auto& ld = data_(ln);
        // patches, cells, patch ghost cells, level ghost cells, particles
        std::array<unsigned long long, 5> mine{}, all{};
        for (std::size_t i = 0; i < damn::amr::size(ld); ++i)
        {
            auto const& box   = ld.patches[i].box;
            auto const& links = l.neighbours.of[i];
            ++mine[0];
            mine[1] += static_cast<unsigned long long>(box.size());
            damn::core::for_each_point(damn::core::grow(box, gw), [&](auto const& p) {
                if (damn::core::isIn(p, box))
                    return;
                bool const neighbour = std::any_of(links.begin(), links.end(), [&](auto const& k) {
                    return damn::core::isIn(p, damn::core::shift(k.patch.box, k.shift));
                });
                ++mine[neighbour ? 2 : 3];
            });
        }
        mine[4] = particle_count(ln);

        all = mine;
        if (n_ranks_ > 1)
            MPI_Reduce(mine.data(), all.data(), static_cast<int>(mine.size()),
                       MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        ss << "lvl:" << ln << " (" << all[0] << "," << all[1] << "," << all[2] << "," << all[3]
           << ") parts(" << all[4] << "), \n";
        total += all[4];
    }
    if (rank_ != 0)
        return "";
    ss << "tot:" << total;
    return ss.str();
}

template<auto opts>
std::string Simulator<opts>::to_str() const
{
    std::stringstream ss;
    ss << "PLAMN SIMULATOR\n";
    ss << "------------------------------------\n";
    ss << "interpolation order  : " << interp_order << "\n";
    ss << "dimension            : " << dimension << "\n";
    ss << "time step            : " << dt_ << "\n";
    ss << "number of time steps : " << timeStepNbr_ << "\n";
    ss << "levels               : " << levels_.size() << "\n";
    return ss.str();
}

} // namespace plamn

#endif // PLAMN_SIMULATOR_SIMULATOR_ACCESS_HPP
