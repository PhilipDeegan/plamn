#ifndef PLAMN_SIMULATOR_SIMULATOR_DIAGNOSTICS_HPP
#define PLAMN_SIMULATOR_SIMULATOR_DIAGNOSTICS_HPP

// plamn::Simulator: diagnostics, phareh5 or pharevtkhdf, a file per quantity; the formats, the
// ranks' order of writes, coupled hierarchies: doc/simulator/simulator_diagnostics.md
// phareh5: every rank creates every patch's datasets (collective), a rank with fewer passes empty
// IWYU pragma: private, include "plamn/simulator/simulator.hpp"

#include "plamn/core/numerics/primitive_conservative/primitive_conservative_transformer.hpp"
#include "damn/amr/hierarchy.hpp"
#include "damn/core/exec/thread_pool.hpp"
#include "damn/core/logger.hpp"
#include "damn/hdf5/damn_hdf5.hpp"

#include "plamn/diagnostic/phareh5.hpp"
#include "plamn/diagnostic/vtkhdf.hpp"
#include "plamn/simulator/simulator.hpp" // IWYU pragma: keep (the class)

#if DAMN_HAS_HIGHFIVE
#include "damn/hdf5/vtkhdf.hpp"
#endif

#include <algorithm>
#include <cmath>
#include <array>
#include <cstddef>
#include <utility>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <cstdint>
#include <type_traits>
#include <span>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace plamn
{
namespace detail
{
    // whether the model's levels have the field quantity
    template<typename Model>
        requires solver::is_mhd_model_v<Model>
    bool provides(Model const& model, std::string const& quantity)
    {
        return model.quantities.contains(quantity);
    }
    template<typename Model>
        requires solver::is_hybrid_model_v<Model>
    bool provides(Model const& model, std::string const& quantity)
    {
        return model.computed.contains(quantity) or model.quantities.contains(quantity);
    }

    // fn(the quantity's fields on a patch of the model's): MHD's V and P (and the bulk velocity,
    //  its V) from the conservative state first (PHARE's compute); a hybrid computed quantity's
    //  scratch grids (computed(patch, quantity): computed per level first), else the view's
    template<typename Model>
        requires solver::is_mhd_model_v<Model>
    void quantity_fields(Model const& model, auto& patch, std::string const& quantity, auto&&,
                         auto&& fn)
    {
        if (quantity == "/mhd/V" or quantity == "/mhd/P" or quantity == "/ions/bulkVelocity")
        {
            auto& u = patch.states[0];
            core::PrimitiveConservativeTransformer{model.gamma}.to_primitive(
                patch.layout, u.rho, u.rhoV, u.B, u.Etot, u.V, u.P);
        }
        fn(diagnostic::quantity_for(model.quantities, quantity)(patch));
    }
    template<typename Model>
        requires solver::is_hybrid_model_v<Model>
    void quantity_fields(Model const& model, auto& patch, std::string const& quantity,
                         auto&& computed, auto&& fn)
    {
        if (model.computed.contains(quantity))
            fn(computed(patch, quantity));
        else
            fn(diagnostic::quantity_for(model.quantities, quantity)(simulator::view(patch)));
    }
} // namespace detail

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
    auto const format = get("format", std::string{"phareh5"});
    if (format != "pharevtkhdf" and format != "phareh5")
        throw std::runtime_error("plamn diagnostics: format " + format
                                 + " not supported (phareh5, pharevtkhdf)");
    diagnostics_.phareh5 = format == "phareh5";
    if (!diagnostics_.phareh5) // pharevtkhdf has no particles (PHARE's neither)
        std::erase_if(diagnostics_.list, [](auto const& d) { return d.type == "particle"; });
    diagnostics_.dir               = get("filePath", std::string{"phare_output"});
    diagnostics_.overwrite         = get("mode", std::string{}) == "overwrite";
    diagnostics_.fine_dump_lvl_max = get("fine_dump_lvl_max", 0);
    if constexpr (Models_t::mhd_on)
        models_.mhd.quantities
            = diagnostic::mhd_quantities<typename solver::MHDModel<opts>::Patch_t>();
    if constexpr (Models_t::hybrid_on)
        models_.hybrid.quantities = solver::hybrid::diagnostic_quantities(models_.hybrid);
    // unknown quantities: at setup, not at a dump. A field on the levels of any model owning
    //  levels that provides it (none: quantity_for's error), particles on the hybrid levels
    bool constexpr mhd_on = Models_t::mhd_on, hybrid_on = Models_t::hybrid_on;
    bool const mhd_levels    = mhd_on and models_.mhd_levels > 0;
    bool const hybrid_levels = hybrid_on and models_.mhd_levels < hierarchy_.max_levels;
    for (auto const& d : diagnostics_.list)
    {
        if (d.type == "particle")
        {
            if constexpr (hybrid_on)
                if (hybrid_levels)
                {
                    std::ignore = solver::hybrid::particle_population(models_.hybrid, d.quantity);
                    continue;
                }
            throw std::runtime_error("plamn diagnostics: particles without hybrid levels: "
                                     + d.quantity);
        }
        bool provided = false;
        if constexpr (mhd_on)
            provided = provided or (mhd_levels and detail::provides(models_.mhd, d.quantity));
        if constexpr (hybrid_on)
            if (hybrid_levels and detail::provides(models_.hybrid, d.quantity))
            {
                provided = true;
                if (is_computed(d.quantity)) // its scratch grids on every patch
                    models_.hybrid.requested.insert(d.quantity);
            }
        if (provided)
            continue;
        if constexpr (hybrid_on)
            if (hybrid_levels)
                std::ignore = diagnostic::quantity_for(models_.hybrid.quantities, d.quantity);
        if constexpr (mhd_on)
            std::ignore = diagnostic::quantity_for(models_.mhd.quantities, d.quantity);
    }
}

template<auto opts>
bool Simulator<opts>::dump_diagnostics(double const timestamp, double const timestep)
{
    bool dumped = false;
    for (auto& diag : diagnostics_.list)
        if (diagnostic::due(diag, timestamp, timestep))
        {
            dump_(diag, timestamp);
            dumped = true;
        }
    return dumped;
}

template<auto opts>
void Simulator<opts>::fine_dump_(std::size_t const ln, double const time)
{
    auto const level0_end = currentTime_ + dt_; // advance() adds dt_ after the levels
    if (ln == 0 or static_cast<int>(ln) > diagnostics_.fine_dump_lvl_max
        or std::abs(time - level0_end) <= 1e-12 * std::max(1., std::abs(level0_end)))
        return;
    for (auto& diag : diagnostics_.list)
        dump_(diag, time, static_cast<int>(ln));
}

template<auto opts>
void Simulator<opts>::dump_(diagnostic::Diagnostic& diag, double const time, int const only)
{
    if (diagnostics_.phareh5)
        dump_phareh5_(diag, time, only);
    else
        dump_vtkhdf_(diag, time, only);
}

template<auto opts>
void Simulator<opts>::dump_vtkhdf_(diagnostic::Diagnostic& diag, double const time, int const only)
{
    DAMN_LOG_SCOPE(1, "Simulator::dump_vtkhdf_");
#if DAMN_HAS_HIGHFIVE
    using namespace damn::hdf5;
    using Float = Diagnostics::Float;

    // fn(the quantity's fields on a patch of the model's)
    auto const computed_   = [&](auto& patch, std::string const& q) { return computed(patch, q); };
    auto const with_fields = [&](auto& model, auto& patch, auto&& fn) {
        detail::quantity_fields(model, patch, diag.quantity, computed_, fn);
    };
    auto const path        = diagnostics_.dir + "/" + diagnostic::file_name(diag.quantity);
    auto const max_levels  = hierarchy_.max_levels;
    std::size_t components = 0;
    {
        // the number of components, from any patch of a level providing it (a rank may have none)
        std::size_t local = 0;
        for (std::size_t ln = 0; ln < levels_.size() and local == 0; ++ln)
            models_.visit(ln, [&](auto& model) {
                auto const& ld = model.levels[static_cast<int>(ln)];
                if (!ld.data.empty() and detail::provides(model, diag.quantity))
                    with_fields(model, *ld.data[0], [&](auto const& f) { local = f.size(); });
            });
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
        if constexpr (Models_t::hybrid_on)
        {
            auto const& ions = dict_["simulation"]["ions"];
            for (std::size_t i = 0; i < models_.hybrid.population_names.size(); ++i)
                if (diag.quantity.starts_with("/ions/pop/" + models_.hybrid.population_names[i]
                                              + "/"))
                {
                    auto const& pop = ions["pop" + std::to_string(i)];
                    file->write_attribute("/", "pop_mass", pop["mass"].template to<double>());
                    file->write_attribute(
                        "/", "pop_charge",
                        pop["particle_initializer"]["charge"].template to<double>());
                }
        }

        std::array<double, dimension> spacing{};
        for (std::size_t i = 0; i < dimension; ++i)
            spacing[i] = hierarchy_.dx[i];
        vtkhdf::init<Float>(*file, components, max_levels, spacing);
    }

    vtkhdf::append_step<Float>(*file, time);
    auto const empty_level = [&](int const ln) {
        damn::amr::Level<dimension> const none{ln, {}};
        vtkhdf::append_level<Float>(*file, ln, none, rank_, components, std::span<Float const>());
    };
    for (int ln = 0; ln < max_levels; ++ln)
    {
        if (static_cast<std::size_t>(ln) >= levels_.size() or (only >= 0 and ln != only))
        {
            empty_level(ln);
            continue;
        }
        models_.visit(static_cast<std::size_t>(ln), [&](auto& model) {
            using Model_t = std::remove_cvref_t<decltype(model)>;
            if (!detail::provides(model, diag.quantity))
                return empty_level(ln); // not on this model's levels: as a level not there
            if constexpr (solver::is_hybrid_model_v<Model_t>)
                if (is_computed(diag.quantity))
                    compute(static_cast<std::size_t>(ln), diag.quantity);
            auto constexpr gw = Model_t::Patch_t::field_ghost_width;
            auto& ld          = model.levels[ln];

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
                        with_fields(model, *ld.data[i], [&](auto const& grids) {
                            for (std::size_t c = 0; c < grids.size(); ++c)
                                diagnostic::primal_into<gw>(*grids[c], ld.patches[i].box, rows,
                                                            components, c);
                        });
                    });
            }
            DAMN_LOG_SCOPE(1, "Simulator::dump_vtkhdf_::write");
            vtkhdf::append_level<Float>(*file, ln, hierarchy_.levels[ln], rank_, components,
                                        std::span<Float const>{buffer.data(), first.back()});
        });
    }

    // PHARE's flush_every: closed every flush_every dumps (0: never), reopened at the next
    if (++diag.dumps; diag.flush_every != 0 and diag.dumps % diag.flush_every == 0)
        diagnostics_.files.erase(path);
#else
    throw std::runtime_error("plamn diagnostics: pharevtkhdf needs HDF5 (damn_hdf5): "
                             + diag.quantity + " at " + std::to_string(time));
#endif
}

template<auto opts>
void Simulator<opts>::dump_phareh5_(diagnostic::Diagnostic& diag, double const time, int const only)
{
    DAMN_LOG_SCOPE(1, "Simulator::dump_phareh5_");
#if DAMN_HAS_HIGHFIVE
    using namespace damn::hdf5;
    namespace ph5     = diagnostic::phareh5;
    using Float       = Diagnostics::Float;

    bool const particles = diag.type == "particle";
    // fn(the quantity's fields on a patch of the model's)
    auto const computed_   = [&](auto& patch, std::string const& q) { return computed(patch, q); };
    auto const with_fields = [&](auto& model, auto& patch, auto&& fn) {
        detail::quantity_fields(model, patch, diag.quantity, computed_, fn);
    };
    // datasets: the quantity's last path element, then _x.. (vectors), _xx.. (tensors)
    auto const name           = diag.quantity.substr(diag.quantity.rfind('/') + 1);
    auto const component_name = [&](std::size_t const c, std::size_t const n) {
        std::array<char const*, 3> constexpr vec{"_x", "_y", "_z"};
        std::array<char const*, 6> constexpr tensor{"_xx", "_xy", "_xz", "_yy", "_yz", "_zz"};
        return n == 1 ? name : name + (n == 3 ? vec.at(c) : tensor.at(c));
    };

    auto const path = diagnostics_.dir + "/" + ph5::file_name(diag.quantity);
    auto& file      = diagnostics_.files[path];
    if (!file)
    {
        bool const truncate = diagnostics_.overwrite and !diagnostics_.created.count(path);
        if (rank_ == 0)
            std::filesystem::create_directories(diagnostics_.dir);
        MPI_Barrier(MPI_COMM_WORLD);
        file = std::make_unique<h5::HighFiveFile>(
            path, truncate ? h5::HiFile::Create | h5::HiFile::Truncate : h5::HiFile::OpenOrCreate);
        diagnostics_.created.insert(path);
    }

    auto const time_key = ph5::time_key(time);
    int const first     = only >= 0 ? only : 0;
    int const last      = only >= 0 ? only : static_cast<int>(levels_.size()) - 1;
    for (int ln = first; ln <= last and static_cast<std::size_t>(ln) < levels_.size(); ++ln)
    {
        models_.visit(static_cast<std::size_t>(ln), [&](auto& model) {
            using Model_t = std::remove_cvref_t<decltype(model)>;
            using Patch_t = Model_t::Patch_t;
            if (!particles and !detail::provides(model, diag.quantity))
                return; // not on this model's levels: nothing, as a level not there
            // the fields' points written: PHARE's ghost box (hybrid: per interp order; MHD: the
            //  patch's)
            auto constexpr gw = Patch_t::field_ghost_width;
            auto constexpr G  = [] {
                if constexpr (solver::is_mhd_model_v<Model_t>)
                    return std::uint32_t{gw};
                else
                    return ph5::field_ghosts<interp_order>;
            }();
            if constexpr (solver::is_hybrid_model_v<Model_t>)
                if (!particles and is_computed(diag.quantity))
                    compute(static_cast<std::size_t>(ln), diag.quantity);
            auto& ld            = model.levels[ln];
            auto const n_local  = ld.data.size();
            auto const n_slots  = damn::mpi::max(n_local);
            auto const patch_at = [&](std::size_t const i) {
                auto const& p = ld.patches[i];
                return ph5::patch_path(time_key, ln, p.rank, p.id);
            };

            if (particles)
            {
                if constexpr (solver::is_hybrid_model_v<Model_t>)
                {
                    auto const pop = solver::hybrid::particle_population(model, diag.quantity);
                    bool const level_ghost = diag.quantity.ends_with("/levelGhost");
                    std::vector<ph5::ParticleColumns<dimension>> columns;
                    for (std::size_t i = 0; i < n_local; ++i)
                    {
                        auto const& p = ld.data[i]->ions[pop];
                        columns.push_back(ph5::particle_columns(
                            level_ghost ? p.levelGhostParticles() : p.domainParticles()));
                    }
                    using Shape = std::vector<std::size_t>;
                    for (std::size_t i = 0; i < n_slots; ++i)
                    {
                        bool const mine     = i < n_local;
                        std::size_t const n = mine ? columns[i].n : 0;
                        auto const at       = mine ? patch_at(i) + "/" : std::string{};
                        auto const shape
                            = [&](std::size_t const cols) { return n ? Shape{n, cols} : Shape{0}; };
                        file->template create_data_set_per_mpi<double>(n ? at + "weight" : "",
                                                                       shape(1));
                        file->template create_data_set_per_mpi<int>(n ? at + "iCell" : "",
                                                                    shape(dimension));
                        file->template create_data_set_per_mpi<double>(n ? at + "delta" : "",
                                                                       shape(dimension));
                        file->template create_data_set_per_mpi<double>(n ? at + "v" : "", shape(3));
                        file->write_attributes_per_mpi(
                            mine ? patch_at(i) : "", "ghosts",
                            std::size_t{ph5::particle_ghosts<interp_order>});
                    }
                    for (std::size_t i = 0; i < n_local; ++i)
                        if (auto const& c = columns[i]; c.n)
                        {
                            auto const at = patch_at(i) + "/";
                            file->write_data_set_flat(at + "weight", c.weight.data());
                            file->write_data_set_flat(at + "iCell", c.iCell.data());
                            file->write_data_set_flat(at + "delta", c.delta.data());
                            file->write_data_set_flat(at + "v", c.v.data());
                        }
                }
            }
            else
            {
                // every component of every local patch, its shape and values
                std::size_t components = 0;
                for (std::size_t i = 0; i < n_local and components == 0; ++i)
                    with_fields(model, *ld.data[i], [&](auto const& f) { components = f.size(); });
                components = damn::mpi::max(components);
                std::vector<std::vector<std::vector<std::size_t>>> shapes(n_local);
                std::vector<std::vector<std::vector<Float>>> values(n_local);
                for (std::size_t i = 0; i < n_local; ++i)
                    with_fields(model, *ld.data[i], [&](auto const& grids) {
                        shapes[i].resize(grids.size());
                        values[i].resize(grids.size());
                        for (std::size_t c = 0; c < grids.size(); ++c)
                            ph5::field_values<gw, G>(*grids[c], ld.patches[i].box, shapes[i][c],
                                                     values[i][c]);
                    });
                for (std::size_t i = 0; i < n_slots; ++i)
                    for (std::size_t c = 0; c < components; ++c)
                    {
                        bool const mine = i < n_local;
                        auto const at   = mine ? patch_at(i) + "/" + component_name(c, components)
                                               : std::string{};
                        file->template create_data_set_per_mpi<Float>(
                            at, mine ? shapes[i][c] : std::vector<std::size_t>(dimension, 0));
                        file->write_attributes_per_mpi(at, "ghosts", std::size_t{G});
                    }
                for (std::size_t i = 0; i < n_local; ++i)
                    for (std::size_t c = 0; c < components; ++c)
                        file->write_data_set_flat(patch_at(i) + "/" + component_name(c, components),
                                                  values[i][c].data());
            }

            // the patches' attributes (PHARE's getPatchProperties)
            for (std::size_t i = 0; i < n_slots; ++i)
            {
                bool const mine = i < n_local;
                auto const at   = mine ? patch_at(i) : std::string{};
                std::vector<int> lower(dimension), upper(dimension);
                std::vector<std::uint32_t> cells(dimension);
                std::vector<double> origin(dimension);
                if (mine)
                {
                    auto const& layout = ld.data[i]->layout;
                    for (std::size_t d = 0; d < dimension; ++d)
                    {
                        lower[d]  = layout.cells.lower[d];
                        upper[d]  = layout.cells.upper[d];
                        cells[d]  = static_cast<std::uint32_t>(upper[d] - lower[d] + 1);
                        origin[d] = layout.origin[d];
                    }
                }
                file->write_attributes_per_mpi(at, "origin", origin);
                file->write_attributes_per_mpi(at, "nbrCells", cells);
                file->write_attributes_per_mpi(at, "lower", lower);
                file->write_attributes_per_mpi(at, "upper", upper);
                file->write_attributes_per_mpi(at, "mpi_rank", static_cast<std::size_t>(rank_));
            }
        });
    }

    // the file's attributes (PHARE's fileAttributes_, the population's, the diagnostic's)
    std::vector<int> domain_upper, periodic;
    std::vector<double> cell_width, origin(dimension, 0.);
    for (std::size_t i = 0; i < dimension; ++i)
    {
        domain_upper.push_back(hierarchy_.domain.upper[i]);
        periodic.push_back(hierarchy_.periodic[i]);
        cell_width.push_back(hierarchy_.dx[i]);
    }
    file->write_attribute("/", "dimension", static_cast<std::size_t>(dimension));
    file->write_attribute("/", "interpOrder", static_cast<std::size_t>(interp_order));
    file->write_attribute("/", "layoutType", std::string{"yee"});
    file->write_attribute("/", "domain_box", domain_upper);
    file->write_attribute("/", "cell_width", cell_width);
    file->write_attribute("/", "origin", origin);
    file->write_attribute("/", "periodicities", periodic);
    if constexpr (Models_t::hybrid_on)
    {
        auto const& ions = dict_["simulation"]["ions"];
        for (std::size_t i = 0; i < models_.hybrid.population_names.size(); ++i)
            if (diag.quantity.starts_with("/ions/pop/" + models_.hybrid.population_names[i] + "/"))
            {
                auto const& pop = ions["pop" + std::to_string(i)];
                file->write_attribute("/", "pop_mass", pop["mass"].template to<double>());
                file->write_attribute("/", "pop_charge",
                                      pop["particle_initializer"]["charge"].template to<double>());
            }
    }
    for (auto const& [key, value] : diag.attributes)
        std::visit([&](auto const& v) { file->write_attribute("/py_attrs", key, v); }, value);

    // PHARE's flush_every: closed every flush_every dumps (0: never), reopened at the next
    if (++diag.dumps; diag.flush_every != 0 and diag.dumps % diag.flush_every == 0)
        diagnostics_.files.erase(path);
#else
    throw std::runtime_error("plamn diagnostics: phareh5 needs HDF5 (damn_hdf5): " + diag.quantity
                             + " at " + std::to_string(time));
#endif
}

} // namespace plamn

#endif // PLAMN_SIMULATOR_SIMULATOR_DIAGNOSTICS_HPP
