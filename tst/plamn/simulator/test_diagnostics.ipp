// pharevtkhdf diagnostics from C++, as PHARE's tests/diagnostic (test-diagnostics.cpp): the
// simulation is a python script (diagnostics_job.py), run by an embedded interpreter through
// pyphare and plamn, which fill plamn's Dict; the Simulator is made from that Dict here, dumps,
// and the files are read back with HighFive and checked against the patches' data:
//
//   every file: PHARE's root attributes, VTKHDF's, one step at t0
//   every level up to max_nbr_levels: its spacing, its boxes (every rank's, rank major), its
//     rows' offsets, and this rank's rows: each primal node of each patch (dual directions
//     averaged, Fortran order, the X_TIMES copies), from the patch's own grids
//   the momentum tensors: the total is the populations' sum
//   an unsupported quantity throws
//
// a binary per dimension (test_diagnostics_<dim>d.cpp defines PLAMN_TEST_DIM, as PHARE's per
// dimension binaries keep a compile's memory down); run from the plamn root, on any number of
// ranks. PYPHARE: the PHARE checkout pyphare comes from (default ~/git/phare/mkn)
//
// the Dict is handed over by address (plamn's Dict.address()): plamn_etc and this binary are
// built from the same headers, the Dict is used in place, the interpreter kept alive meanwhile
// (its init functions are python callables)

#include <Python.h> // first, as python asks

#include "damn/core/geometry/box.hpp"
#include "damn/core/geometry/point.hpp"
#include "damn/hdf5/vtkhdf.hpp"

#include "plamn/core/data/grid/layout_defs.hpp"
#include "plamn/simulator/simulator.hpp"

#include "highfive/H5File.hpp"

#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <system_error>
#include <tuple>
#include <type_traits>
#include <vector>

namespace
{
auto constexpr dim  = std::size_t{PLAMN_TEST_DIM};
auto constexpr opts = plamn::SimOpts{.dimension     = dim,
                                     .interp_order  = 1,
                                     .nbRefinedPart = std::array<std::size_t, 3>{2, 4, 6}[dim - 1],
                                     .layout_mode   = plamn::core::LayoutMode::AoSMapped,
                                     .alloc_mode    = damn::AllocatorMode::CPU};
using Simulator     = plamn::Simulator<opts>;
using Patch         = plamn::solver::HybridModel<opts>::Patch_t;
using Grid          = plamn::solver::HybridModel<opts>::Grid_t;
using Box           = damn::core::Box<int, dim>;
using Point         = damn::core::Point<int, dim>;

// diagnostics_job.py's populations: name, mass, charge
std::vector<std::tuple<std::string, double, double>> const populations{{"protons", 1., 1.},
                                                                       {"alpha", 4., 2.}};

plamn::initializer::Dict* dict = nullptr; // plamn's, filled by the job
std::string out_dir;

int rank()
{
    int r = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &r);
    return r;
}

// a quantity's grids on a patch, components in file order, straight from the patch's storage
std::vector<Grid*> grids_of(Simulator& sim, Patch& patch, std::string const& quantity)
{
    auto const all = [](auto& grids) {
        std::vector<Grid*> out;
        for (auto& g : grids)
            out.push_back(std::addressof(g));
        return out;
    };
    if (quantity == "/EM_B")
        return all(patch.emB);
    if (quantity == "/EM_E")
        return all(patch.emE);
    if (quantity == "/ions/charge_density")
        return {std::addressof(patch.chargeDensity)};
    if (quantity == "/ions/mass_density")
        return {std::addressof(patch.massDensity)};
    if (quantity == "/ions/bulkVelocity")
        return all(patch.bulkVelocity);
    if (quantity == "/ions/charge_flux")
        return all(patch.chargeFlux);
    for (std::size_t i = 0; i < populations.size(); ++i)
    {
        auto const pop = "/ions/pop/" + std::get<0>(populations[i]) + "/";
        if (quantity == pop + "density")
            return {std::addressof(patch.populations[i]->density)};
        if (quantity == pop + "charge_density")
            return {std::addressof(patch.populations[i]->charge_density)};
        if (quantity == pop + "flux")
            return all(patch.populations[i]->flux);
    }
    return sim.computed(patch, quantity); // momentum tensors
}

std::vector<std::string> quantities()
{
    std::vector<std::string> out{"/EM_B",
                                 "/EM_E",
                                 "/ions/charge_density",
                                 "/ions/mass_density",
                                 "/ions/bulkVelocity",
                                 "/ions/charge_flux",
                                 "/ions/momentum_tensor"};
    for (auto const& [name, mass, charge] : populations)
        for (auto const* q : {"density", "charge_density", "flux", "momentum_tensor"})
            out.push_back("/ions/pop/" + name + "/" + q);
    return out;
}

std::string file_of(std::string const& quantity)
{
    auto name = quantity.substr(1);
    std::replace(name.begin(), name.end(), '/', '_');
    return out_dir + "/" + name + ".vtkhdf";
}

// the grid's value on a primal node: the mean of its dual neighbours, per dual direction
double expected(Grid const& grid, Box const& cells, Point const& node)
{
    auto const centering = plamn::core::centering<dim>(grid);
    auto constexpr gw    = Patch::field_ghost_width;
    double sum           = 0;
    std::size_t n        = 0;
    for (std::uint32_t corner = 0; corner < (1u << dim); ++corner)
    {
        Point p    = node;
        bool valid = true;
        for (std::size_t i = 0; i < dim; ++i)
        {
            bool const high = (corner >> i) & 1u;
            if (centering[i] == plamn::core::QtyCentering::primal)
                valid = valid and !high;
            else if (!high)
                p[i] -= 1; // the dual cells node - 1 and node
        }
        if (!valid)
            continue;
        sum += grid(plamn::core::amr_to_local(cells, p, gw).toArray());
        ++n;
    }
    return sum / static_cast<double>(n);
}

// one file against the simulator
void check_file(Simulator& sim, std::string const& quantity)
{
    SCOPED_TRACE(quantity);
    auto const max_levels = sim.hierarchy().max_levels;

    // the momentum tensors' scratch grids hold the dump's values once computed again
    if (sim.is_computed(quantity))
        for (std::size_t ln = 0; ln < sim.n_levels(); ++ln)
            sim.compute(ln, quantity);

    HighFive::File file{file_of(quantity), HighFive::File::ReadOnly};
    auto const root = file.getGroup("/");
    EXPECT_EQ(root.getAttribute("dimension").read<std::uint64_t>(), dim);
    EXPECT_EQ(root.getAttribute("interpOrder").read<std::uint64_t>(), 1u);
    std::vector<int> const periodic(dim, 1);
    EXPECT_EQ(root.getAttribute("periodicities").read<std::vector<int>>(), periodic);
    std::vector<int> upper;
    for (std::size_t i = 0; i < dim; ++i)
        upper.push_back(sim.hierarchy().domain.upper[i]);
    EXPECT_EQ(root.getAttribute("domain_box").read<std::vector<int>>(), upper);
    for (auto const& [name, mass, charge] : populations)
        if (quantity.starts_with("/ions/pop/" + name + "/"))
        {
            EXPECT_DOUBLE_EQ(root.getAttribute("pop_mass").read<double>(), mass);
            EXPECT_DOUBLE_EQ(root.getAttribute("pop_charge").read<double>(), charge);
        }

    auto const vtk = file.getGroup("/VTKHDF");
    EXPECT_EQ(vtk.getAttribute("Version").read<std::vector<int>>(), (std::vector<int>{2, 2}));
    EXPECT_EQ(vtk.getAttribute("Type").read<std::string>(), "OverlappingAMR");
    EXPECT_EQ(file.getGroup("/VTKHDF/Steps").getAttribute("NSteps").read<int>(), 1);
    EXPECT_EQ(file.getDataSet("/VTKHDF/Steps/Values").read<std::vector<float>>(),
              std::vector<float>{0.f});

    for (int ln = 0; ln < max_levels; ++ln)
    {
        SCOPED_TRACE("level " + std::to_string(ln));
        auto const level = "/VTKHDF/Level" + std::to_string(ln);
        auto const steps = "/VTKHDF/Steps/Level" + std::to_string(ln);

        auto const spacing
            = file.getGroup(level).getAttribute("Spacing").read<std::vector<float>>();
        ASSERT_EQ(spacing.size(), dim);
        for (std::size_t i = 0; i < dim; ++i)
            EXPECT_FLOAT_EQ(spacing[i],
                            static_cast<float>(sim.hierarchy().dx[i] / std::pow(2, ln)));

        auto const n_boxes
            = file.getDataSet(steps + "/NumberOfAMRBox").read<std::vector<std::int64_t>>();
        auto const box_off
            = file.getDataSet(steps + "/AMRBoxOffset").read<std::vector<std::int64_t>>();
        auto const row_off
            = file.getDataSet(steps + "/PointDataOffset/data").read<std::vector<std::int64_t>>();
        ASSERT_EQ(n_boxes.size(), 1u);
        EXPECT_EQ(box_off[0], 0);
        EXPECT_EQ(row_off[0], 0);

        bool const exists = static_cast<std::size_t>(ln) < sim.n_levels();
        using Patches     = std::remove_cvref_t<decltype(sim.hierarchy().levels[0].patches)>;
        auto patches      = exists ? sim.hierarchy().levels[ln].patches : Patches{};
        std::stable_sort(patches.begin(), patches.end(),
                         [](auto const& a, auto const& b) { return a.rank < b.rank; });
        ASSERT_EQ(n_boxes[0], static_cast<std::int64_t>(patches.size()));

        // every rank's boxes, rank major
        auto const boxes = file.getDataSet(level + "/AMRBox").read<std::vector<std::vector<int>>>();
        ASSERT_EQ(boxes.size(), patches.size());
        std::size_t rows = 0, first_mine = 0;
        for (std::size_t k = 0; k < patches.size(); ++k)
        {
            for (std::size_t i = 0; i < 3; ++i)
            {
                EXPECT_EQ(boxes[k][2 * i], i < dim ? patches[k].box.lower[i] : 0);
                EXPECT_EQ(boxes[k][2 * i + 1], i < dim ? patches[k].box.upper[i] : 0);
            }
            if (patches[k].rank < rank())
                first_mine += damn::hdf5::vtkhdf::rows_of(patches[k].box);
            rows += damn::hdf5::vtkhdf::rows_of(patches[k].box);
        }

        auto const data
            = file.getDataSet(level + "/PointData/data").read<std::vector<std::vector<float>>>();
        ASSERT_EQ(data.size(), rows);
        if (!exists)
            continue;

        // this rank's patches, in level order, from first_mine
        auto& ld        = sim.level(ln);
        std::size_t row = first_mine;
        for (std::size_t p = 0; p < ld.patches.size(); ++p)
        {
            auto const& cells = ld.patches[p].box;
            auto const grids  = grids_of(sim, *ld.data[p], quantity);
            ASSERT_EQ(data[row].size(), grids.size());

            std::array<std::size_t, dim> shape{};
            std::size_t points = 1;
            for (std::size_t i = 0; i < dim; ++i)
                points *= shape[i] = static_cast<std::size_t>(cells.upper[i] - cells.lower[i] + 2);

            std::size_t bad = 0;
            for (std::size_t local = 0; local < points; ++local)
            {
                Point node; // x fastest
                std::size_t rest = local;
                for (std::size_t i = 0; i < dim; ++i)
                {
                    node[i] = cells.lower[i] + static_cast<int>(rest % shape[i]);
                    rest /= shape[i];
                }
                for (std::size_t c = 0; c < grids.size(); ++c)
                {
                    auto const want = expected(*grids[c], cells, node);
                    for (std::size_t copy = 0; copy < damn::hdf5::vtkhdf::x_times<dim>; ++copy)
                    {
                        auto const got = data[row + copy * points + local][c];
                        if (std::abs(got - want) > 1e-6 * std::max(1., std::abs(want))
                            and bad++ < 5)
                            ADD_FAILURE() << "patch " << p << " node " << local << " component "
                                          << c << " copy " << copy << ": " << got << " != " << want;
                    }
                }
            }
            EXPECT_EQ(bad, 0u);
            row += damn::hdf5::vtkhdf::rows_of(cells);
        }
    }
}

} // namespace


TEST(SimulatorDiagnosticsTest, allFromPython)
{
    Simulator sim{*dict};
    sim.initialize();
    EXPECT_TRUE(sim.dump_diagnostics(/*timestamp=*/0, sim.timeStep()));
    MPI_Barrier(MPI_COMM_WORLD); // every rank's rows written (the files closed: flush_every 1)

    ASSERT_EQ(sim.n_levels(), 2u);
    for (auto const& quantity : quantities())
        check_file(sim, quantity);

    // the total momentum tensor: the populations' sum
    for (std::size_t ln = 0; ln < sim.n_levels(); ++ln)
    {
        std::vector<std::string> keys{"/ions/momentum_tensor"};
        for (auto const& [name, mass, charge] : populations)
            keys.push_back("/ions/pop/" + name + "/momentum_tensor");
        for (auto const& key : keys)
            sim.compute(ln, key);
        for (auto& patch : sim.level(ln).data)
        {
            auto const total = sim.computed(*patch, keys[0]);
            for (std::size_t c = 0; c < 6; ++c)
            {
                auto const& t = *total[c];
                for (std::size_t i = 0; i < t.size(); ++i)
                {
                    double sum = 0;
                    for (std::size_t k = 1; k < keys.size(); ++k)
                        sum += sim.computed(*patch, keys[k])[c]->data()[i];
                    EXPECT_NEAR(t.data()[i], sum, 1e-12 * std::max(1., std::abs(sum)));
                }
            }
        }
    }
}

TEST(SimulatorDiagnosticsTest, unsupportedQuantityThrows)
{
    auto copy = *dict;
    cppdict::add("simulation/diagnostics/fluid/fluid0/quantity",
                 std::string{"/ions/pressure_tensor"}, copy);
    EXPECT_THROW(Simulator{copy}, std::runtime_error);
}


int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    out_dir = "phare_outputs/plamn_test_diagnostics_" + std::to_string(dim) + "d/"
              + std::to_string(size); // concurrent runs on other rank counts
    setenv("PLAMN_TEST_DIM", std::to_string(dim).c_str(), 1);
    setenv("PLAMN_DIAGS_TEST_DIR", out_dir.c_str(), 1);

    Py_Initialize();
    int result = 1;
    {
        auto const job = R"(
import os, sys
sys.path[:0] = [".", os.environ.get("PYPHARE", os.path.expanduser("~/git/phare/mkn"))
                + "/pyphare", "tst/plamn/simulator"]
import diagnostics_job
diagnostics_job.declare()
import pyphare.pharein as ph
ph.populateDict()
import plamn
plamn_dict = plamn.current_dict().address()
)";
        if (PyRun_SimpleString(job) != 0)
            throw std::runtime_error("diagnostics_job.py failed");
        auto* main_ = PyImport_AddModule("__main__");
        auto* addr  = PyObject_GetAttrString(main_, "plamn_dict");
        dict        = reinterpret_cast<plamn::initializer::Dict*>(PyLong_AsVoidPtr(addr));
        Py_DECREF(addr);

        result = RUN_ALL_TESTS();
    }
    // the Dict's python functions go with the interpreter (plamn's atexit clear_dict)
    Py_Finalize();

    // a failed run's files kept to look at, as PHARE
    MPI_Barrier(MPI_COMM_WORLD);
    if (result == 0 and rank() == 0)
    {
        std::filesystem::remove_all(out_dir);
        std::error_code ec; // the binary's directory too, once no other rank count uses it
        std::filesystem::remove(std::filesystem::path{out_dir}.parent_path(), ec);
    }
    MPI_Finalize();
    return result;
}
