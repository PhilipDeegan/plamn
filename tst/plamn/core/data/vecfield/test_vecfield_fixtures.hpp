#ifndef PLAMN_TEST_CORE_DATA_TEST_VECFIELD_FIXTURES_HPP
#define PLAMN_TEST_CORE_DATA_TEST_VECFIELD_FIXTURES_HPP

#include "plamn/core/def.hpp"

#include "damn/core/def/damn_config.hpp"
#include "plamn/core/data/vecfield/vecfield.hpp"
#include "plamn/core/data/particles/particles_def.hpp"

#include "tst/plamn/core/data/field/test_field_fixtures.hpp"
#include "tst/plamn/core/data/tensorfield/test_tensorfield_fixtures.hpp"

namespace plamn::core
{


template<auto opts>
using UsableVecField = UsableTensorField<opts.with_rank(1)>;


} // namespace plamn::core

#endif /*PLAMN_TEST_CORE_DATA_TEST_VECFIELD_FIXTURES_HPP*/
