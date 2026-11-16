#ifndef PLAMN_SIMULATOR_SIMULATOR_DIAGNOSTICS_HPP
#define PLAMN_SIMULATOR_SIMULATOR_DIAGNOSTICS_HPP

// diagnostics (pyphare's diag_options and *Diagnostics, simulation/diagnostics), written as
// pharevtkhdf (PHARE's format): a file per quantity, a step per dump, every level up to
// max_nbr_levels (an empty entry while a level is not there)
//
// a dump of a quantity, level by level: this rank's patches' rows (damn::hdf5::vtkhdf::rows_of)
// in one buffer (diagnostics_.buffer, reused), each patch converted straight into its slice
// (plamn::diagnostic::primal_into), the patches in parallel (damn's thread pool), then one
// write per rank (damn::hdf5::vtkhdf::append_level). Scopes: dump, convert, write per level
// IWYU pragma: private, include "plamn/simulator/simulator.hpp"

#include "damn/amr/hierarchy.hpp"
#include "damn/core/exec/thread_pool.hpp"
#include "damn/core/logger.hpp"
#include "damn/hdf5/damn_hdf5.hpp"

#include "plamn/diagnostic/vtkhdf.hpp"
#include "plamn/simulator/simulator.hpp" // IWYU pragma: keep (the class)

#if DAMN_HAS_HIGHFIVE
#include "damn/hdf5/vtkhdf.hpp"
#endif

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace plamn
{
template<auto opts>
void Simulator<opts>::setup_diagnostics_()
{
    diagnostics_.list
        = ::plamn::diagnostic::diagnostics_from(dict_); // a member is named diagnostic
    if (diagnostics_.list.empty())
        return;
    auto const get = [&](std::string const& key, auto const fallback) {
        return cppdict::get_value(dict_, "simulation/diagnostics/" + key, fallback);
    };
    if (auto const format = get("format", std::string{"phareh5"}); format != "pharevtkhdf")
        throw std::runtime_error("plamn diagnostics: format " + format
                                 + " not supported (pharevtkhdf)");
    diagnostics_.dir        = get("filePath", std::string{"phare_output"});
    diagnostics_.overwrite  = get("mode", std::string{}) == "overwrite";
    diagnostics_.quantities = ::plamn::diagnostic::hybrid_quantities<View_t>(population_names_);
    for (auto const& d : diagnostics_.list) // unknown quantities: at setup, not at a dump
        if (!is_computed(d.quantity))
            std::ignore = diagnostic::quantity_for(diagnostics_.quantities, d.quantity);
}

template<auto opts>
bool Simulator<opts>::dump_diagnostics(double const timestamp, double const timestep)
{
    bool dumped = false;
    for (auto& diag : diagnostics_.list)
        if (diagnostic::due(diag, timestamp, timestep))
        {
            dump_vtkhdf_(diag, timestamp);
            dumped = true;
        }
    return dumped;
}

template<auto opts>
void Simulator<opts>::dump_vtkhdf_(diagnostic::Diagnostic& diag, double const time)
{
    DAMN_LOG_SCOPE(1, "Simulator::dump_vtkhdf_");
#if DAMN_HAS_HIGHFIVE
    using namespace damn::hdf5;
    using Float       = Diagnostics::Float;
    auto constexpr gw = Patch_t::field_ghost_width;

    // the quantity's fields on a patch, fn(fields): computed ones' scratch grids (computed per
    //  level first), the others the view's
    bool const is_computed_ = is_computed(diag.quantity);
    auto const with_fields  = [&](Patch_t& patch, auto&& fn) {
        if (is_computed_)
            fn(computed(patch, diag.quantity));
        else
            fn(diagnostic::quantity_for(diagnostics_.quantities,
                                        diag.quantity)(simulator::view(patch)));
    };
    auto const path        = diagnostics_.dir + "/" + diagnostic::file_name(diag.quantity);
    auto const max_levels  = hierarchy_.max_levels;
    std::size_t components = 0;
    {
        // the number of components, from any patch's accessor (a rank may have none)
        std::size_t local = 0;
        for (auto const& l : levels_)
            if (!l.data.data.empty())
            {
                with_fields(*l.data.data[0], [&](auto const& f) { local = f.size(); });
                break;
            }
        components = damn::mpi::max(local);
    }

    auto& file = diagnostics_.files[path];
    if (!file)
    {
        bool const truncate = diagnostics_.overwrite and !diagnostics_.created.count(path);
        if (rank_ == 0)
            std::filesystem::create_directories(diagnostics_.dir);
        MPI_Barrier(MPI_COMM_WORLD);
        file = std::make_unique<h5::HighFiveFile>(
            path, truncate ? h5::HiFile::Create | h5::HiFile::Truncate : h5::HiFile::OpenOrCreate);
        diagnostics_.created.insert(path);

        // PHARE's root attributes
        file->write_attribute("/", "dimension", static_cast<std::uint64_t>(dimension));
        file->write_attribute("/", "interpOrder", static_cast<std::uint64_t>(interp_order));
        std::vector<int> upper, periodic;
        for (std::size_t i = 0; i < dimension; ++i)
        {
            upper.push_back(hierarchy_.domain.upper[i]);
            periodic.push_back(hierarchy_.periodic[i]);
        }
        file->write_attribute("/", "domain_box", upper);
        file->write_attribute("/", "periodicities", periodic);
        auto const& ions = dict_["simulation"]["ions"];
        for (std::size_t i = 0; i < population_names_.size(); ++i)
            if (diag.quantity.starts_with("/ions/pop/" + population_names_[i] + "/"))
            {
                auto const& pop = ions["pop" + std::to_string(i)];
                file->write_attribute("/", "pop_mass", pop["mass"].template to<double>());
                file->write_attribute("/", "pop_charge",
                                      pop["particle_initializer"]["charge"].template to<double>());
            }

        std::array<double, dimension> spacing{};
        for (std::size_t i = 0; i < dimension; ++i)
            spacing[i] = hierarchy_.dx[i];
        vtkhdf::init<Float>(*file, components, max_levels, spacing);
    }

    vtkhdf::append_step<Float>(*file, time);
    for (int ln = 0; ln < max_levels; ++ln)
    {
        if (static_cast<std::size_t>(ln) >= levels_.size())
        {
            damn::amr::Level<dimension> const none{ln, {}};
            vtkhdf::append_level<Float>(*file, ln, none, rank_, components,
                                        std::span<Float const>());
            continue;
        }
        if (is_computed_)
            compute(static_cast<std::size_t>(ln), diag.quantity);
        auto& ld = levels_[ln].data;

        // this rank's patches' rows, one after another
        std::vector<std::size_t> first{0};
        for (auto const& p : ld.patches)
            first.push_back(first.back() + vtkhdf::rows_of(p.box) * components);
        auto& buffer = diagnostics_.buffer.get(first.back());
        {
            DAMN_LOG_SCOPE(1, "Simulator::dump_vtkhdf_::convert");
            damn::core::ThreadPool::INSTANCE().patch_exec(
                ld.data.size(), [&](auto const i) { return i; },
                [&](auto const i) {
                    std::span<Float> rows{buffer.data() + first[i], first[i + 1] - first[i]};
                    with_fields(*ld.data[i], [&](auto const& grids) {
                        for (std::size_t c = 0; c < grids.size(); ++c)
                            diagnostic::primal_into<gw>(*grids[c], ld.patches[i].box, rows,
                                                        components, c);
                    });
                });
        }
        DAMN_LOG_SCOPE(1, "Simulator::dump_vtkhdf_::write");
        vtkhdf::append_level<Float>(*file, ln, hierarchy_.levels[ln], rank_, components,
                                    std::span<Float const>{buffer.data(), first.back()});
    }

    // PHARE's flush_every: closed every flush_every dumps (0: never), reopened at the next
    if (++diag.dumps; diag.flush_every != 0 and diag.dumps % diag.flush_every == 0)
        diagnostics_.files.erase(path);
#else
    throw std::runtime_error("plamn diagnostics: pharevtkhdf needs HDF5 (damn_hdf5): "
                             + diag.quantity + " at " + std::to_string(time));
#endif
}

} // namespace plamn

#endif // PLAMN_SIMULATOR_SIMULATOR_DIAGNOSTICS_HPP
