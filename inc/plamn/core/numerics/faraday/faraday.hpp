#ifndef PLAMN_FARADAY_HPP
#define PLAMN_FARADAY_HPP

#include "damn/core/def/damn_config.hpp"

namespace plamn::core {

// Bnew = B - dt curl(E), one component at one point: curlEc is that component of curl(layout, E)
//  where and how it is evaluated (boxes, tiles, GPU) is up to the caller,
//  see core/numerics/faraday/faraday_transformer.hpp
struct Faraday {
  void operator()(auto const& ijk, auto& Bnc, auto const& Bc, auto const& curlEc,
                  double const dt) const DAMN_ALL {
    Bnc(ijk) = Bc(ijk) - dt * curlEc(ijk);
  }
};

}  // namespace plamn::core

#endif
