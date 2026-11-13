#ifndef PLAMN_CORE_INCLUDE_HPP
#define PLAMN_CORE_INCLUDE_HPP

#include "plamn/core/def.hpp"

#include "damn/core/def/damn_config.hpp" // IWYU pragma: keep

#include "plamn/core/data/grid/grid.hpp"
#include "plamn/core/data/ions/ions.hpp"
#include "plamn/core/data/grid/grid_tiles.hpp"
#include "plamn/core/data/grid/gridlayout.hpp"
#include "plamn/core/data/vecfield/vecfield.hpp"
#include "plamn/core/data/electrons/electrons.hpp"
#include "plamn/core/data/electromag/electromag.hpp"
#include "plamn/core/models/options/mhd_options.hpp"
#include "damn/core/data/ndarray/ndarray_vector.hpp"
#include "plamn/core/data/particles/particles.hpp"
#include "plamn/core/models/options/hybrid_options.hpp"
#include "plamn/core/data/ions/ion_population/ion_population.hpp"
#include "plamn/core/data/ions/particle_initializers/particle_initializer_factory.hpp"
#include "plamn/core/data/ions/particle_initializers/maxwellian_particle_initializer.hpp"

#include "dict.hpp"

#include <string>
#include <cstddef>
#include <functional>
#include <string_view>
#include <unordered_map>

namespace plamn::core
{

using field_value_type = double;


template<typename GridLayout_t, auto L_, auto A_>
struct UsingResolver
{
    bool static constexpr c_ordering = true;
    auto static constexpr dimension  = GridLayout_t::dimension;

    using Scalar  = decltype(GridLayout_t::options)::FieldOptions::Scalar;
    using Array_t = damn::core::NdArrayVector<dimension, field_value_type, c_ordering, A_>;
    using Grid_t  = Grid<Array_t, Scalar>;
    using Field_t = Field<dimension, Scalar, field_value_type, A_>;
};

template<typename GridLayout_t, auto L_, auto A_>
    requires(is_tiled(L_))
struct UsingResolver<GridLayout_t, L_, A_>
{
    bool static constexpr c_ordering = true;
    auto static constexpr dimension  = GridLayout_t::dimension;

    using Scalar          = decltype(GridLayout_t::options)::FieldOptions::Scalar;
    using base_types      = UsingResolver<GridLayout_t, LayoutMode::AoS, A_>;
    using Grid_base_type  = base_types::Grid_t;
    using Field_base_type = basic::Field<FieldOpts<Scalar, field_value_type>{dimension, A_}>;
    using Grid_t          = GridTileSet<GridLayout_t, Grid_base_type, Field_base_type>;
    using Field_t         = FieldTileSet<GridLayout_t, Grid_base_type, Field_base_type>;
};


template<auto opts_>
struct PLAMN_Types
{
    // auto static constexpr opts           = opts_;
    auto static constexpr dimension      = opts_.dimension;
    auto static constexpr layout_mode    = opts_.layout_mode;
    auto static constexpr allocator_mode = opts_.alloc_mode;

    struct Hybrid
    {
        auto static constexpr opts           = opts_;
        auto static constexpr field_options  = HybridFieldOptions<opts>{};
        auto static constexpr hybrid_options = HybridOptions<field_options>{};
        using GridLayout_t                   = GridLayout<hybrid_options>;

        using Resolver_t = UsingResolver<GridLayout_t, layout_mode, allocator_mode>;

        using Array_t
            = damn::core::NdArrayVector<dimension, field_value_type, Resolver_t::c_ordering, allocator_mode>;

        using Grid_t  = Resolver_t::Grid_t;  // layout dependent
        using Field_t = Resolver_t::Field_t; // layout dependent

        using VecField_t       = VecField<Field_t, HybridQuantity>;
        using SymTensorField_t = SymTensorField<Field_t, HybridQuantity>;
        using Electromag_t     = Electromag<VecField_t>;

        using Particle_t = Particle<dimension>;

        using Particles_t = Particles<ParticlesOptions{
            dimension, layout_mode, StorageMode::VECTOR, allocator_mode}>;

        using ParticleAoS_t = Particles_t;
        using ParticleSoA_t = ParticlesSoA<dimension>;

        using MaxwellianParticleInitializer_t
            = MaxwellianParticleInitializer<Particles_t, GridLayout_t>;
        using IonPopulation_t = IonPopulation<Particles_t, VecField_t, SymTensorField_t>;
        using Ions_t          = Ions<IonPopulation_t, GridLayout_t>;
        using Electrons_t     = Electrons<Ions_t>;

        using ParticleInitializerFactory_t
            = ParticleInitializerFactory<Particles_t, GridLayout_t>;
    };


    struct MHD
    {
        auto static constexpr opts          = opts_;
        auto static constexpr field_options = MHDFieldOptions<opts>{};
        auto static constexpr mhd_options   = MHDOptions<field_options>{};
        using GridLayout_t                  = GridLayout<mhd_options>;
        using Resolver_t = UsingResolver<GridLayout_t, layout_mode, allocator_mode>;

        using Array_t
            = damn::core::NdArrayVector<dimension, field_value_type, Resolver_t::c_ordering, allocator_mode>;

        using Grid_t           = Resolver_t::Grid_t;  // layout dependent
        using Field_t          = Resolver_t::Field_t; // layout dependent
        using VecField_t       = VecField<Field_t, MHDQuantity>;
        using SymTensorField_t = SymTensorField<Field_t, MHDQuantity>;
        using Electromag_t     = Electromag<VecField_t>;
    };
};

struct PLAMN_Sim_Types
{
    using SimFunctorParams = cppdict::Dict<int, unsigned int, double, std::size_t>;
    using SimFunctor       = std::function<void(SimFunctorParams const& /*params*/)>;
    using SimulationFunctors // code place id -> function_id -> function
        = std::unordered_map<std::string, std::unordered_map<std::string, SimFunctor>>;
};


} // namespace plamn::core


namespace plamn::strings
{

constexpr static std::string_view cma = ",";
}

#endif // PLAMN_CORE_INCLUDE_HPP
