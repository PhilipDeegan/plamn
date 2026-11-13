#include "plamn/core/def.hpp"

#include "plamn/plamn_core.hpp"
#include "plamn/simulator/simulator_def.hpp"

#include "damn/core/geometry/point.hpp"
#include "damn/core/geometry/point.hpp"
#include "plamn/core/data/grid/layout_defs.hpp"
#include "plamn/core/numerics/faraday/faraday_transformer.hpp"

#include "plamn/simulator/simulator_def.hpp"

#include "tst/plamn/core/data/field/test_field.hpp"
#include "tst/plamn/core/data/vecfield/test_vecfield_fixtures.hpp"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <fstream>


using namespace plamn::core;





std::vector<double> read(std::string filename)
{
    std::ifstream readFile(filename);
    assert(readFile.is_open());
    std::vector<double> x;

    std::copy(std::istream_iterator<double>(readFile), std::istream_iterator<double>(),
              std::back_inserter(x));
    return x;
}



class Faraday1DTest : public ::testing::Test
{
protected:
    static constexpr auto dim          = 1;
    static constexpr auto interp_order = 1;

    using core_types                 = plamn::core::PLAMN_Types<plamn::SimOpts{dim, interp_order}>;
    using HybridTypes                = core_types::Hybrid;
    using Layout_t               = HybridTypes::Layout_t;
    using Faraday_t                  = FaradayTransformer;
    auto static constexpr field_opts = plamn::core::TensorFieldOptions<HybridTypes>{};
    using UsableVecFieldND           = UsableVecField<field_opts>;

    Layout_t layout;

    UsableVecFieldND B, E, Bnew;

public:
    Faraday1DTest()
        : layout{{{0.1}}, {{50}}, damn::core::Point{0.}}
        , B{"B", layout, HybridQuantity::Vector::B}
        , E{"E", layout, HybridQuantity::Vector::E}
        , Bnew{"Bnew", layout, HybridQuantity::Vector::B}
    {
    }
};




class Faraday2DTest : public ::testing::Test
{
protected:
    static constexpr auto dim          = 2;
    static constexpr auto interp_order = 1;

    using core_types                 = plamn::core::PLAMN_Types<plamn::SimOpts{dim, interp_order}>;
    using HybridTypes                = core_types::Hybrid;
    using Layout_t               = HybridTypes::Layout_t;
    using Faraday_t                  = FaradayTransformer;
    auto static constexpr field_opts = plamn::core::TensorFieldOptions<HybridTypes>{};
    using UsableVecFieldND           = UsableVecField<field_opts>;

    Layout_t layout;
    UsableVecFieldND B, E, Bnew;

public:
    Faraday2DTest()
        : layout{{{0.1, 0.2}}, {{50, 30}}, damn::core::Point{0., 0.}}
        , B{"B", layout, HybridQuantity::Vector::B}
        , E{"E", layout, HybridQuantity::Vector::E}
        , Bnew{"Bnew", layout, HybridQuantity::Vector::B}
    {
    }
};




class Faraday3DTest : public ::testing::Test
{
protected:
    static constexpr auto dim          = 3;
    static constexpr auto interp_order = 1;

    using core_types                 = plamn::core::PLAMN_Types<plamn::SimOpts{dim, interp_order}>;
    using HybridTypes                = core_types::Hybrid;
    using Layout_t               = HybridTypes::Layout_t;
    using Faraday_t                  = FaradayTransformer;
    auto static constexpr field_opts = plamn::core::TensorFieldOptions<HybridTypes>{};
    using UsableVecFieldND           = UsableVecField<field_opts>;

    Layout_t layout;
    UsableVecFieldND B, E, Bnew;

public:
    Faraday3DTest()
        : layout{{{0.1, 0.2, 0.3}}, {{50, 30, 40}}, damn::core::Point{0., 0., 0.}}
        , B{"B", layout, HybridQuantity::Vector::B}
        , E{"E", layout, HybridQuantity::Vector::E}
        , Bnew{"Bnew", layout, HybridQuantity::Vector::B}
    {
    }
};




TEST_F(Faraday1DTest, Faraday1DCalculatedOk)
{
    auto filename_dbydt = std::string{"dbydt_yee_1D_order1.txt"};
    auto filename_dbzdt = std::string{"dbzdt_yee_1D_order1.txt"};
    auto expected_dbydt = read(filename_dbydt);
    auto expected_dbzdt = read(filename_dbzdt);

    auto gsi_p_X = this->layout.ghostStartIndex(QtyCentering::primal, Direction::X);
    auto gei_p_X = this->layout.ghostEndIndex(QtyCentering::primal, Direction::X);

    auto& [Ex, Ey, Ez]                = E();
    auto& [Bx, By, Bz]                = B();
    auto const& [Bxnew, Bynew, Bznew] = Bnew();

    for (auto ix = gsi_p_X; ix <= gei_p_X; ++ix)
    {
        auto point
            = coordinate(this->layout, Ey, this->layout.localToAMR(damn::core::Point{ix}.as_signed()));

        Ey(ix) = std::cos(2 * M_PI / 5. * point[0]);
        Ez(ix) = std::sin(2 * M_PI / 5. * point[0]);
    }

    auto gsi_d_X = this->layout.ghostStartIndex(QtyCentering::dual, Direction::X);
    auto gei_d_X = this->layout.ghostEndIndex(QtyCentering::dual, Direction::X);

    for (auto ix = gsi_d_X; ix <= gei_d_X; ++ix)
    {
        auto point
            = coordinate(this->layout, By, this->layout.localToAMR(damn::core::Point{ix}.as_signed()));

        By(ix) = std::tanh(point[0] - 5. / 2.);
        Bz(ix) = std::tanh(point[0] - 5. / 2.);
    }

    Faraday_t{}(layout, B, E, Bnew, 1.);

    auto psi_d_X = this->layout.physicalStartIndex(QtyCentering::dual, Direction::X);
    auto pei_d_X = this->layout.physicalEndIndex(QtyCentering::dual, Direction::X);

    for (auto ix = psi_d_X; ix <= pei_d_X; ++ix)
    {
        EXPECT_THAT(Bynew(ix), ::testing::DoubleNear((expected_dbydt[ix]), 1e-12));
        EXPECT_THAT(Bznew(ix), ::testing::DoubleNear((expected_dbzdt[ix]), 1e-12));
    }
}




TEST_F(Faraday2DTest, Faraday2DCalculatedOk)
{
    auto filename_dbxdt = std::string{"dbxdt_yee_2D_order1.txt"};
    auto filename_dbydt = std::string{"dbydt_yee_2D_order1.txt"};
    auto filename_dbzdt = std::string{"dbzdt_yee_2D_order1.txt"};
    auto expected_dbxdt = read(filename_dbxdt);
    auto expected_dbydt = read(filename_dbydt);
    auto expected_dbzdt = read(filename_dbzdt);

    auto gsi_p_X = this->layout.ghostStartIndex(QtyCentering::primal, Direction::X);
    auto gei_p_X = this->layout.ghostEndIndex(QtyCentering::primal, Direction::X);
    auto gsi_d_X = this->layout.ghostStartIndex(QtyCentering::dual, Direction::X);
    auto gei_d_X = this->layout.ghostEndIndex(QtyCentering::dual, Direction::X);

    auto gsi_p_Y = this->layout.ghostStartIndex(QtyCentering::primal, Direction::Y);
    auto gei_p_Y = this->layout.ghostEndIndex(QtyCentering::primal, Direction::Y);
    auto gsi_d_Y = this->layout.ghostStartIndex(QtyCentering::dual, Direction::Y);
    auto gei_d_Y = this->layout.ghostEndIndex(QtyCentering::dual, Direction::Y);

    auto& [Ex, Ey, Ez]                = E();
    auto& [Bx, By, Bz]                = B();
    auto const& [Bxnew, Bynew, Bznew] = Bnew();

    for (auto ix = gsi_d_X; ix <= gei_d_X; ++ix)
    {
        for (auto iy = gsi_p_Y; iy <= gei_p_Y; ++iy)
        {
            auto point = coordinate(this->layout, 
                Ex, this->layout.localToAMR(damn::core::Point{ix, iy}.as_signed()));

            Ex(ix, iy) = std::cos(2 * M_PI / 5. * point[0]) * std::sin(2 * M_PI / 6. * point[1]);
        }
    }

    for (auto ix = gsi_p_X; ix <= gei_p_X; ++ix)
    {
        for (auto iy = gsi_d_Y; iy <= gei_d_Y; ++iy)
        {
            auto point = coordinate(this->layout, 
                Ey, this->layout.localToAMR(damn::core::Point{ix, iy}.as_signed()));

            Ey(ix, iy) = std::cos(2 * M_PI / 5. * point[0]) * std::tanh(2 * M_PI / 6. * point[1]);
        }
    }

    for (auto ix = gsi_p_X; ix <= gei_p_X; ++ix)
    {
        for (auto iy = gsi_p_Y; iy <= gei_p_Y; ++iy)
        {
            auto point = coordinate(this->layout, 
                Ez, this->layout.localToAMR(damn::core::Point{ix, iy}.as_signed()));

            Ez(ix, iy) = std::sin(2 * M_PI / 5. * point[0]) * std::tanh(2 * M_PI / 6. * point[1]);
        }
    }

    for (auto ix = gsi_p_X; ix <= gei_p_X; ++ix)
    {
        for (auto iy = gsi_d_Y; iy <= gei_d_Y; ++iy)
        {
            auto point = coordinate(this->layout, 
                Bx, this->layout.localToAMR(damn::core::Point{ix, iy}.as_signed()));

            Bx(ix, iy) = std::tanh(point[0] - 5. / 2.) * std::tanh(point[1] - 6. / 2.);
        }
    }

    for (auto ix = gsi_d_X; ix <= gei_d_X; ++ix)
    {
        for (auto iy = gsi_p_Y; iy <= gei_p_Y; ++iy)
        {
            auto point = coordinate(this->layout, 
                By, this->layout.localToAMR(damn::core::Point{ix, iy}.as_signed()));

            By(ix, iy) = std::tanh(point[0] - 5. / 2.) * std::tanh(point[1] - 6. / 2.);
        }
    }

    for (auto ix = gsi_d_X; ix <= gei_d_X; ++ix)
    {
        for (auto iy = gsi_d_Y; iy <= gei_d_Y; ++iy)
        {
            auto point = coordinate(this->layout, 
                Bz, this->layout.localToAMR(damn::core::Point{ix, iy}.as_signed()));

            Bz(ix, iy) = std::tanh(point[0] - 5. / 2.) * std::tanh(point[1] - 6. / 2.);
        }
    }

    Faraday_t{}(layout, B, E, Bnew, 1.);

    auto psi_p_X = this->layout.physicalStartIndex(QtyCentering::primal, Direction::X);
    auto pei_p_X = this->layout.physicalEndIndex(QtyCentering::primal, Direction::X);
    auto psi_d_Y = this->layout.physicalStartIndex(QtyCentering::dual, Direction::Y);
    auto pei_d_Y = this->layout.physicalEndIndex(QtyCentering::dual, Direction::Y);

    auto nPts_ = alloc_shape(this->layout, HybridQuantity::Scalar::Bx);

    for (auto ix = psi_p_X; ix <= pei_p_X; ++ix)
    {
        for (auto iy = psi_d_Y; iy <= pei_d_Y; ++iy)
        {
            auto index_ = ix * nPts_[1] + iy;
            EXPECT_THAT(Bxnew(ix, iy), ::testing::DoubleNear((expected_dbxdt[index_]), 1e-12));
        }
    }

    auto psi_d_X = this->layout.physicalStartIndex(QtyCentering::dual, Direction::X);
    auto pei_d_X = this->layout.physicalEndIndex(QtyCentering::dual, Direction::X);
    auto psi_p_Y = this->layout.physicalStartIndex(QtyCentering::primal, Direction::Y);
    auto pei_p_Y = this->layout.physicalEndIndex(QtyCentering::primal, Direction::Y);

    nPts_ = alloc_shape(this->layout, HybridQuantity::Scalar::By);

    for (auto ix = psi_d_X; ix <= pei_d_X; ++ix)
    {
        for (auto iy = psi_p_Y; iy <= pei_p_Y; ++iy)
        {
            auto index_ = ix * nPts_[1] + iy;
            EXPECT_THAT(Bynew(ix, iy), ::testing::DoubleNear((expected_dbydt[index_]), 1e-12));
        }
    }

    nPts_ = alloc_shape(this->layout, HybridQuantity::Scalar::Bz);

    for (auto ix = psi_d_X; ix <= pei_d_X; ++ix)
    {
        for (auto iy = psi_d_Y; iy <= pei_d_Y; ++iy)
        {
            auto index_ = ix * nPts_[1] + iy;
            EXPECT_THAT(Bznew(ix, iy), ::testing::DoubleNear((expected_dbzdt[index_]), 1e-12));
        }
    }
}



TEST_F(Faraday3DTest, Faraday3DCalculatedOk)
{
    auto filename_dbxdt = std::string{"dbxdt_yee_3D_order1.txt"};
    auto filename_dbydt = std::string{"dbydt_yee_3D_order1.txt"};
    auto filename_dbzdt = std::string{"dbzdt_yee_3D_order1.txt"};
    auto expected_dbxdt = read(filename_dbxdt);
    auto expected_dbydt = read(filename_dbydt);
    auto expected_dbzdt = read(filename_dbzdt);

    auto gsi_p_X = this->layout.ghostStartIndex(QtyCentering::primal, Direction::X);
    auto gei_p_X = this->layout.ghostEndIndex(QtyCentering::primal, Direction::X);
    auto gsi_d_X = this->layout.ghostStartIndex(QtyCentering::dual, Direction::X);
    auto gei_d_X = this->layout.ghostEndIndex(QtyCentering::dual, Direction::X);

    auto gsi_p_Y = this->layout.ghostStartIndex(QtyCentering::primal, Direction::Y);
    auto gei_p_Y = this->layout.ghostEndIndex(QtyCentering::primal, Direction::Y);
    auto gsi_d_Y = this->layout.ghostStartIndex(QtyCentering::dual, Direction::Y);
    auto gei_d_Y = this->layout.ghostEndIndex(QtyCentering::dual, Direction::Y);

    auto gsi_p_Z = this->layout.ghostStartIndex(QtyCentering::primal, Direction::Z);
    auto gei_p_Z = this->layout.ghostEndIndex(QtyCentering::primal, Direction::Z);
    auto gsi_d_Z = this->layout.ghostStartIndex(QtyCentering::dual, Direction::Z);
    auto gei_d_Z = this->layout.ghostEndIndex(QtyCentering::dual, Direction::Z);

    auto& [Ex, Ey, Ez]                = E();
    auto& [Bx, By, Bz]                = B();
    auto const& [Bxnew, Bynew, Bznew] = Bnew();

    for (auto ix = gsi_d_X; ix <= gei_d_X; ++ix)
    {
        for (auto iy = gsi_p_Y; iy <= gei_p_Y; ++iy)
        {
            for (auto iz = gsi_p_Z; iz <= gei_p_Z; ++iz)
            {
                auto point = coordinate(this->layout, 
                    Ex, this->layout.localToAMR(damn::core::Point{ix, iy, iz}.as_signed()));

                Ex(ix, iy, iz) = std::sin(2 * M_PI / 5. * point[0])
                                 * std::cos(2 * M_PI / 6. * point[1])
                                 * std::tanh(2 * M_PI / 12. * point[2]);
            }
        }
    }

    for (auto ix = gsi_p_X; ix <= gei_p_X; ++ix)
    {
        for (auto iy = gsi_d_Y; iy <= gei_d_Y; ++iy)
        {
            for (auto iz = gsi_p_Z; iz <= gei_p_Z; ++iz)
            {
                auto point = coordinate(this->layout, 
                    Ey, this->layout.localToAMR(damn::core::Point{ix, iy, iz}.as_signed()));

                Ey(ix, iy, iz) = std::tanh(2 * M_PI / 5. * point[0])
                                 * std::sin(2 * M_PI / 6. * point[1])
                                 * std::cos(2 * M_PI / 12. * point[2]);
            }
        }
    }

    for (auto ix = gsi_p_X; ix <= gei_p_X; ++ix)
    {
        for (auto iy = gsi_p_Y; iy <= gei_p_Y; ++iy)
        {
            for (auto iz = gsi_d_Z; iz <= gei_d_Z; ++iz)
            {
                auto point = coordinate(this->layout, 
                    Ez, this->layout.localToAMR(damn::core::Point{ix, iy, iz}.as_signed()));

                Ez(ix, iy, iz) = std::cos(2 * M_PI / 5. * point[0])
                                 * std::tanh(2 * M_PI / 6. * point[1])
                                 * std::sin(2 * M_PI / 12. * point[2]);
            }
        }
    }

    for (auto ix = gsi_p_X; ix <= gei_p_X; ++ix)
    {
        for (auto iy = gsi_d_Y; iy <= gei_d_Y; ++iy)
        {
            for (auto iz = gsi_d_Z; iz <= gei_d_Z; ++iz)
            {
                auto point = coordinate(this->layout, 
                    Bx, this->layout.localToAMR(damn::core::Point{ix, iy, iz}.as_signed()));

                Bx(ix, iy, iz) = std::tanh(point[0] - 5. / 2.) * std::tanh(point[1] - 6. / 2.)
                                 * std::tanh(point[2] - 12. / 2.);
            }
        }
    }

    for (auto ix = gsi_d_X; ix <= gei_d_X; ++ix)
    {
        for (auto iy = gsi_p_Y; iy <= gei_p_Y; ++iy)
        {
            for (auto iz = gsi_d_Z; iz <= gei_d_Z; ++iz)
            {
                auto point = coordinate(this->layout, 
                    By, this->layout.localToAMR(damn::core::Point{ix, iy, iz}.as_signed()));

                By(ix, iy, iz) = std::tanh(point[0] - 5. / 2.) * std::tanh(point[1] - 6. / 2.)
                                 * std::tanh(point[2] - 12. / 2.);
            }
        }
    }

    for (auto ix = gsi_d_X; ix <= gei_d_X; ++ix)
    {
        for (auto iy = gsi_d_Y; iy <= gei_d_Y; ++iy)
        {
            for (auto iz = gsi_p_Z; iz <= gei_p_Z; ++iz)
            {
                auto point = coordinate(this->layout, 
                    Bz, this->layout.localToAMR(damn::core::Point{ix, iy, iz}.as_signed()));

                Bz(ix, iy, iz) = std::tanh(point[0] - 5. / 2.) * std::tanh(point[1] - 6. / 2.)
                                 * std::tanh(point[2] - 12. / 2.);
            }
        }
    }

    Faraday_t{}(layout, B, E, Bnew, 1.);

    auto psi_p_X = this->layout.physicalStartIndex(QtyCentering::primal, Direction::X);
    auto pei_p_X = this->layout.physicalEndIndex(QtyCentering::primal, Direction::X);
    auto psi_d_X = this->layout.physicalStartIndex(QtyCentering::dual, Direction::X);
    auto pei_d_X = this->layout.physicalEndIndex(QtyCentering::dual, Direction::X);

    auto psi_p_Y = this->layout.physicalStartIndex(QtyCentering::primal, Direction::Y);
    auto pei_p_Y = this->layout.physicalEndIndex(QtyCentering::primal, Direction::Y);
    auto psi_d_Y = this->layout.physicalStartIndex(QtyCentering::dual, Direction::Y);
    auto pei_d_Y = this->layout.physicalEndIndex(QtyCentering::dual, Direction::Y);

    auto psi_p_Z = this->layout.physicalStartIndex(QtyCentering::primal, Direction::Z);
    auto pei_p_Z = this->layout.physicalEndIndex(QtyCentering::primal, Direction::Z);
    auto psi_d_Z = this->layout.physicalStartIndex(QtyCentering::dual, Direction::Z);
    auto pei_d_Z = this->layout.physicalEndIndex(QtyCentering::dual, Direction::Z);

    auto nPts_ = alloc_shape(this->layout, HybridQuantity::Scalar::Bx);

    for (auto ix = psi_p_X; ix <= pei_p_X; ++ix)
    {
        for (auto iy = psi_d_Y; iy <= pei_d_Y; ++iy)
        {
            for (auto iz = psi_d_Z; iz <= pei_d_Z; ++iz)
            {
                auto index_ = ix * nPts_[1] * nPts_[2] + iy * nPts_[2] + iz;
                EXPECT_THAT(Bxnew(ix, iy, iz),
                            ::testing::DoubleNear((expected_dbxdt[index_]), 1e-12));
            }
        }
    }

    nPts_ = alloc_shape(this->layout, HybridQuantity::Scalar::By);

    for (auto ix = psi_d_X; ix <= pei_d_X; ++ix)
    {
        for (auto iy = psi_p_Y; iy <= pei_p_Y; ++iy)
        {
            for (auto iz = psi_d_Z; iz <= pei_d_Z; ++iz)
            {
                auto index_ = ix * nPts_[1] * nPts_[2] + iy * nPts_[2] + iz;
                EXPECT_THAT(Bynew(ix, iy, iz),
                            ::testing::DoubleNear((expected_dbydt[index_]), 1e-12));
            }
        }
    }

    nPts_ = alloc_shape(this->layout, HybridQuantity::Scalar::Bz);

    for (auto ix = psi_d_X; ix <= pei_d_X; ++ix)
    {
        for (auto iy = psi_d_Y; iy <= pei_d_Y; ++iy)
        {
            for (auto iz = psi_p_Z; iz <= pei_p_Z; ++iz)
            {
                auto index_ = ix * nPts_[1] * nPts_[2] + iy * nPts_[2] + iz;
                EXPECT_THAT(Bznew(ix, iy, iz),
                            ::testing::DoubleNear((expected_dbzdt[index_]), 1e-12));
            }
        }
    }
}




int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
