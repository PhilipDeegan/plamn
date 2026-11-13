#ifndef PLAMN_RECONSTRUCTION_NGHOSTS_HPP
#define PLAMN_RECONSTRUCTION_NGHOSTS_HPP

#include "plamn/core/def.hpp"
#include "plamn/core/models/options/mhd_options_def.hpp"

#include <cstdint>

namespace plamn::MHDOpts
{

// Compile-time mapping from ReconstructionType enum to stencil width (nghosts).
// These values correspond to the nghosts member in each reconstruction class.
template<ReconstructionType R>
struct ReconstructionNghosts;

template<>
struct ReconstructionNghosts<ReconstructionType::Constant>
{
    static constexpr std::uint32_t value = 1;
};

template<>
struct ReconstructionNghosts<ReconstructionType::Linear>
{
    static constexpr std::uint32_t value = 2;
};

template<>
struct ReconstructionNghosts<ReconstructionType::WENO3>
{
    static constexpr std::uint32_t value = 2;
};

template<>
struct ReconstructionNghosts<ReconstructionType::WENOZ>
{
    static constexpr std::uint32_t value = 3;
};

template<>
struct ReconstructionNghosts<ReconstructionType::MP5>
{
    static constexpr std::uint32_t value = 3;
};

template<ReconstructionType R>
inline constexpr std::uint32_t reconstruction_nghosts_v = ReconstructionNghosts<R>::value;

} // namespace plamn::MHDOpts

#endif // PLAMN_RECONSTRUCTION_NGHOSTS_HPP
