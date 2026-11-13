#ifndef PLAMN_TEST_CORE_DATA_ION_POPULATIONS_ION_POPULATION_FIXTURES_HPP
#define PLAMN_TEST_CORE_DATA_ION_POPULATIONS_ION_POPULATION_FIXTURES_HPP

#include "plamn/core/def.hpp"


#include "plamn/plamn_core.hpp"
#include "plamn/core/data/ions/ions.hpp"
#include "damn/core/data/ndarray/ndarray_vector.hpp"
#include "plamn/core/data/particles/particles.hpp"
#include "plamn/core/data/ions/ion_population/ion_population.hpp"

#include "tst/plamn/core/data/particles/test_particles.hpp"
#include "tst/plamn/core/data/grid/test_layout_fixtures.hpp"
#include "tst/plamn/core/data/vecfield/test_vecfield_fixtures.hpp"
#include "tst/plamn/core/data/particles/test_particles_fixtures.hpp"


#include <cassert>

namespace plamn::core
{


template<auto opts>
struct UsableIonsDefaultTypes
{
public:
    using PLAMN_Types     = plamn::core::PLAMN_Types<opts>;
    using Hybrid          = PLAMN_Types::Hybrid;
    using Layout_t        = Hybrid::Layout_t;
    using Particles_t = Hybrid::Particles_t;
    using Quantity        = HybridQuantity;

    auto constexpr static field_opts = TensorFieldOptions<Hybrid>{};

    using UsableVecField_t    = UsableVecField<field_opts>;
    using UsableTensorField_t = UsableTensorField<field_opts>;

    using Grid_t        = UsableTensorField_t::Grid_t;
    using VecField_t    = UsableVecField_t::Super;
    using TensorField_t = UsableTensorField_t::Super;

    using IonPopulation_t = IonPopulation<Particles_t, VecField_t, TensorField_t>;
};


auto inline pop_dict(std::string const& name, std::size_t const ppc = 0)
{
    plamn::initializer::Dict popdict;
    popdict["name"]                           = name;
    popdict["mass"]                           = 1.0;
    auto particle_initializer                 = plamn::initializer::Dict{};
    particle_initializer["nbr_part_per_cell"] = static_cast<int>(ppc);
    particle_initializer["charge"]            = 1.;
    particle_initializer["basis"]             = std::string{"cartesian"};
    popdict["particle_initializer"]           = particle_initializer;
    return popdict;
}

template<typename _defaults>
class UsableIonsPopulation : public _defaults::IonPopulation_t
{
    using Quantity        = _defaults::Quantity;
    using Layout_t        = _defaults::Layout_t;
    using VecField_t      = _defaults::VecField_t;
    using TensorField_t   = _defaults::TensorField_t;
    using Particles_t = _defaults::Particles_t;
    using Super           = IonPopulation<Particles_t, VecField_t, TensorField_t>;



    void set()
    {
        auto&& [_F, _M, _pd, _cd, _particles] = Super::getCompileTimeResourcesViewList();
        F.set_on(_F);
        M.set_on(_M);
        _pd.setBuffer(&rhoP);
        _cd.setBuffer(&rhoC);
        _particles.setBuffer(&particles.pack());

        // DAMN_LOG_LINE_SS(_pd.data()[0] << " == " << _pd.data()[_pd.size() - 1]);
        assert(compare_fields(_pd, *rhoP));
    }


public:
    UsableIonsPopulation(plamn::initializer::Dict const& dict, Layout_t const& layout)
        : Super{dict}
        , layout_{layout}
        , rhoP{this->name() + "_particleDensity", layout_, Quantity::Scalar::rho, 1}
        , rhoC{this->name() + "_chargeDensity", layout_, Quantity::Scalar::rho, 1}
        , F{this->name() + "_flux", layout, Quantity::Vector::V, 1}
        , M{this->name() + "_momentumTensor", layout, Quantity::Tensor::M, 1}
        , particles{this->name(), layout}
    {
        set();
    }

    UsableIonsPopulation(UsableIonsPopulation const& that)
        : Super{pop_dict(that.name())}
        , layout_{that.layout_}
        , rhoP{that.rhoP}
        , rhoC{that.rhoC}
        , F{that.F}
        , M{that.M}
        , particles{that.particles}
    {
        set();
    }

    UsableIonsPopulation(UsableIonsPopulation&&)                 = default;
    UsableIonsPopulation& operator=(UsableIonsPopulation const&) = delete;
    UsableIonsPopulation& operator=(UsableIonsPopulation&&)      = default;


    Super& super() { return *this; }
    Super const& super() const { return *this; }
    auto& operator*() { return super(); }
    auto& operator*() const { return super(); }

    Layout_t layout_;
    _defaults::Grid_t rhoP, rhoC;
    _defaults::UsableVecField_t F;
    _defaults::UsableTensorField_t M;
    UsableParticlesPopulation<Particles_t> particles;
};


template<typename _defaults>
class UsableIons : public Ions<typename _defaults::IonPopulation_t, typename _defaults::Layout_t>
{
    using Quantity     = _defaults::Quantity;
    using Layout_t     = _defaults::Layout_t;
    using Super        = Ions<typename _defaults::IonPopulation_t, Layout_t>;



    auto static super(auto const& pop_names, std::size_t const ppc)
    {
        plamn::initializer::Dict dict;
        dict["nbrPopulations"] = pop_names.size();
        for (std::size_t i = 0; i < pop_names.size(); ++i)
            dict["pop" + std::to_string(i)] = pop_dict(pop_names[i], ppc);
        return dict;
    }



    auto static super(Super const supe)
    {
        plamn::initializer::Dict dict;
        dict["nbrPopulations"] = supe.size();
        for (std::size_t i = 0; i < supe.size(); ++i)
            dict["pop" + std::to_string(i)] = pop_dict(supe[i].name());
        return dict;
    }

    void set()
    {
        auto&& [_bV, _M, _cd, _md, _cF] = Super::getCompileTimeResourcesViewList();
        Vi.set_on(_bV);
        chargeFlux.set_on(_cF);
        M.set_on(_M);
        _cd.setBuffer(&rhoC);
        _md.setBuffer(&rhoM);

        auto& super_pops = Super::getRunTimeResourcesViewList();
        super_pops.clear();
        for (auto& pop : populations)
            super_pops.emplace_back(*pop);
    }


public:
    using Particles_t = typename _defaults::Particles_t;

    UsableIons(Layout_t const& layout, plamn::initializer::Dict const& dict)
        : Super{dict}
        , layout_{layout}
        , rhoM{"massDensity", layout_, Quantity::Scalar::rho, 1}
        , rhoC{"chargeDensity", layout_, Quantity::Scalar::rho, 1}
        , Vi{"bulkVel", layout, Quantity::Vector::V, 1}
        , chargeFlux{"chargeFlux", layout, Quantity::Vector::V, 1}
        , M{"momentumTensor", layout, Quantity::Tensor::M, 0}
    {
        auto& super_pops = Super::getRunTimeResourcesViewList();
        populations.reserve(super_pops.size());
        for (std::size_t i = 0; i < super_pops.size(); ++i)
            populations.emplace_back(dict["pop" + std::to_string(i)], layout);
        set();
    }

    UsableIons(Layout_t const& layout, std::vector<std::string> const& pop_names,
               std::size_t const ppc = 0)
        : UsableIons{layout, super(pop_names, ppc)}
    {
    }

    UsableIons(Layout_t const& layout, std::string const& pop_name, std::size_t const ppc = 0)
        : UsableIons{layout, std::vector<std::string>{pop_name}, ppc}
    {
    }

    UsableIons(UsableIons&& that)
        : Super(super(*that))
        , layout_{that.layout_}
        , rhoM{std::move(that.rhoM)}
        , rhoC{std::move(that.rhoC)}
        , Vi{std::move(that.Vi)}
        , chargeFlux{std::move(that.chargeFlux)}
        , M{std::move(that.M)}
        , populations{std::move(that.populations)}
    {
        set();
    }

    UsableIons(UsableIons const& that)
        : Super(super(*that))
        , layout_{that.layout_}
        , rhoM{that.rhoM}
        , rhoC{that.rhoC}
        , Vi{that.Vi}
        , chargeFlux{that.chargeFlux}
        , M{that.M}
        , populations{that.populations}
    {
        set();
    }

    UsableIons& operator=(UsableIons const&) = delete;
    UsableIons& operator=(UsableIons&&)      = default;

    Super& super() { return *this; }
    Super const& super() const { return *this; }
    auto& operator*() { return super(); }
    auto& operator*() const { return super(); }

    Layout_t layout_;
    _defaults::Grid_t rhoM, rhoC;
    _defaults::UsableVecField_t Vi, chargeFlux;
    _defaults::UsableTensorField_t M;
    std::vector<UsableIonsPopulation<_defaults>> populations;
};

template<auto opts>
using UsableIonsPopulation_t = UsableIonsPopulation<UsableIonsDefaultTypes<opts>>;


template<auto opts>
using UsableIons_t = UsableIons<UsableIonsDefaultTypes<opts>>;



} // namespace plamn::core


#endif /* PLAMN_TEST_CORE_DATA_ION_POPULATIONS_ION_POPULATION_FIXTURES_HPP */
