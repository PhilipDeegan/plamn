#ifndef PLAMN_SIMULATOR_RESTART_PATHS_HPP
#define PLAMN_SIMULATOR_RESTART_PATHS_HPP

// where restarts are: <dir>/<time>/restart.h5, the time with 10 digits each side of the point
// (pyphare's directory names, exact enough to tell steps apart, sorted as strings)

#include "plamn/core/def.hpp"

#include <format>
#include <string>

namespace plamn::simulator
{
MKN_KUL_FN_SIG_RET inline std::string restart_dir_for_time(std::string const& dir,
                                                           double const time)
{
    return std::format("{}/{:021.10f}", dir, time);
}

MKN_KUL_FN_SIG_RET inline std::string restart_file_in(std::string const& dir)
{
    return dir + "/restart.h5";
}
} // namespace plamn::simulator

#endif // PLAMN_SIMULATOR_RESTART_PATHS_HPP
