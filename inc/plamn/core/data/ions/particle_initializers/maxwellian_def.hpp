#ifndef PLAMN_CORE_DATA_IONS_PARTICLE_INITIALIZERS_MAXWELLIAN_DEF_HPP
#define PLAMN_CORE_DATA_IONS_PARTICLE_INITIALIZERS_MAXWELLIAN_DEF_HPP

#include "mkn/kul/math.hpp"
#include "plamn/core/def.hpp"

#include <array>
#include <cstdint>

namespace plamn::core
{
MKN_KUL_FN_SIG_RET inline std::array<double, 3>
basisTransform(std::array<std::array<double, 3>, 3> const& basis, std::array<double, 3> const& vec)
{
    std::array<double, 3> newVec;

    for (std::uint32_t comp = 0; comp < 3; comp++)
        newVec[comp] = basis[0][comp] * vec[0] + basis[1][comp] * vec[1] + basis[2][comp] * vec[2];

    return newVec;
}


// orthonormal basis with the first vector along B, identity if B ~ 0
inline void localMagneticBasis(std::array<double, 3> B, std::array<std::array<double, 3>, 3>& basis)
{
    auto b2 = mkn::kul::math::norm(B);

    if (b2 < 1e-8)
    {
        basis = {{{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}}};
        return;
    }

    basis[0] = {B[0] / b2, B[1] / b2, B[2] / b2};

    // (1,1,1) x B
    basis[1]     = {B[2] - B[1], B[0] - B[2], B[1] - B[0]};
    auto vecNorm = mkn::kul::math::norm(basis[1]);
    for (auto& c : basis[1])
        c /= vecNorm;

    basis[2] = {basis[0][1] * basis[1][2] - basis[0][2] * basis[1][1],
                basis[0][2] * basis[1][0] - basis[0][0] * basis[1][2],
                basis[0][0] * basis[1][1] - basis[0][1] * basis[1][0]};
}

} // namespace plamn::core

#endif // PLAMN_CORE_DATA_IONS_PARTICLE_INITIALIZERS_MAXWELLIAN_DEF_HPP
