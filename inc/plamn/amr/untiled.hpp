#ifndef PLAMN_AMR_UNTILED_HPP
#define PLAMN_AMR_UNTILED_HPP

// untiled temporaries of a patch's tiled fields, for what works on whole patches (damn's
//  schedules): doc/amr/untiled.md
// a lease sets every value of its temporary, nothing carries over from an earlier use (no NaN
//  into a sum)

#include "plamn/amr/resources.hpp"
#include "plamn/core/data/field/field_box.hpp"
#include "plamn/core/data/field/field_tiles.hpp"
#include "plamn/core/data/grid/patch_layout.hpp"
#include "mkn/kul/lang/cpp/compiler/def.hpp"

#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <cstddef>
#include <format>
#include <vector>

namespace plamn::amr
{
enum class Lease : std::uint8_t { read, fill, sum };

template<typename Tiled>
class Untiled
{
public:
    using grid_type  = Tiled::grid_type;      // owns a temporary's memory: the pool only
    using field_type = grid_type::field_type; // a view of it: what leases hand out
    using pq_type    = Tiled::physical_quantity_type;

    field_type& lease(std::string const& name, Tiled& tiled, auto const& layout, Lease const how)
    {
        if (auto* const leased = find(name))
            return *leased; // leased once for this operation (a name both src and dst)

        auto const pq = tiled.physicalQuantity();
        auto& pool    = pools_[pq];
        auto& used    = used_[pq];
        if (used == pool.size())
            pool.push_back(std::make_unique<grid_type>(name, layout, pq));
        auto& tmp = **pool[used++]; // the grid's view, no reallocation

        if (how == Lease::sum)
            core::reduce_into(tmp, tiled); // zeroed, then every tile added
        else
            core::copy_fields(tmp, tiled); // zeroed, then every tile's values
        if (n_active_ == active_.size())
            active_.emplace_back();
        auto& a      = active_[n_active_++];
        a.name       = name; // the kept string's storage
        a.tiled      = std::addressof(tiled);
        a.tmp        = std::addressof(tmp);
        a.write_back = how != Lease::read;
        return tmp;
    }

    field_type* find(std::string const& name) const
    {
        for (std::size_t i = 0; i < n_active_; ++i)
            if (active_[i].name == name)
                return active_[i].tmp;
        return nullptr;
    }

    MKN_KUL_FN_SIG_RET bool leased() const { return n_active_ > 0; }

    void release()
    {
        for (std::size_t i = 0; i < n_active_; ++i)
            if (auto const& a = active_[i]; a.write_back)
                core::copy_fields(*a.tiled, *a.tmp); // every tile's ghost box
        n_active_ = 0;
        for (auto& entry : used_) // the entries kept: leased again every step
            entry.second = 0;
    }

private:
    struct Active
    {
        std::string name;
        Tiled* tiled    = nullptr;
        field_type* tmp = nullptr;
        bool write_back = false;
    };

    std::map<pq_type, std::vector<std::unique_ptr<grid_type>>> pools_;
    std::map<pq_type, std::size_t> used_;
    std::vector<Active> active_; // [0, n_active_): this operation's, the rest kept for reuse
    std::size_t n_active_ = 0;
};

// a patch's untiled temporaries when its grids are tiled, else a placeholder that holds none
template<typename Grid_t>
struct NotTiled
{
    using field_type = Grid_t;
    Grid_t* find(std::string const&) const { return nullptr; }
    bool leased() const { return false; }
    void release() {}
};
template<typename Grid_t>
using UntiledFor_t
    = std::conditional_t<core::is_field_tile_set_v<Grid_t>, Untiled<Grid_t>, NotTiled<Grid_t>>;

// fn(an untiled field of field's values): the field itself when it is not tiled, else a
//  temporary of its tiles' values, for an occasional whole patch read (diagnostics)
template<core::TiledField Field>
void with_untiled(Field const& field, auto&& fn)
{
    using Grid_t      = typename Field::grid_type;
    auto const layout = core::sub_layout(field()[0].layout(), field.box());
    Grid_t tmp{field.name(), layout, field.physicalQuantity()};
    core::copy_fields(*tmp, field);
    fn(std::as_const(*tmp));
}
void with_untiled(auto const& field, auto&& fn)
    requires(!core::TiledField<decltype(field)>)
{
    fn(field);
}

// fn(an untiled field of field's values) then those values back into field: a tiled field's
//  every tile ghost box (outer and inner ghosts), an untiled field is passed itself
template<core::TiledField Field>
void modify_untiled(Field& field, auto&& fn)
{
    using Grid_t      = typename Field::grid_type;
    auto const layout = core::sub_layout(field()[0].layout(), field.box());
    Grid_t tmp{field.name(), layout, field.physicalQuantity()};
    core::copy_fields(*tmp, field);
    fn(*tmp);
    core::copy_fields(field, *tmp);
}
void modify_untiled(auto& field, auto&& fn)
    requires(!core::TiledField<decltype(field)>)
{
    fn(field);
}

// fn(name, the field resource damn sees for a patch's grid): a tiled grid's leased untiled
//  temporary (none outside a lease), an untiled grid itself
template<std::uint32_t gw>
void field_resource_for(auto const& untiled, std::string const& name, core::TiledField auto&,
                        auto&& fn)
{
    using Field_t = typename std::decay_t<decltype(untiled)>::field_type;
    auto r        = FieldResource<Field_t, gw>{untiled.find(name)};
    fn(name, r);
}
template<std::uint32_t gw>
void field_resource_for(auto const&, std::string const& name, auto& grid, auto&& fn)
    requires(!core::TiledField<decltype(grid)>)
{
    auto r = field_resource<gw>(grid);
    fn(name, r);
}

// fn(std::array{&rx, &ry, &rz}): field resources over a vector's three component grids, untiled
//  (tiled ones: copies of their tiles, written back into every tile ghost box after fn)
template<std::uint32_t gw>
void with_untiled_resources(auto& grids, auto&& fn)
{
    modify_untiled(grids[0], [&](auto& x) {
        modify_untiled(grids[1], [&](auto& y) {
            modify_untiled(grids[2], [&](auto& z) {
                auto rx = field_resource<gw>(x);
                auto ry = field_resource<gw>(y);
                auto rz = field_resource<gw>(z);
                fn(std::array{&rx, &ry, &rz});
            });
        });
    });
}

} // namespace plamn::amr

#endif // PLAMN_AMR_UNTILED_HPP
