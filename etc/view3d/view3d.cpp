// view3d: a live 3D view of a running simulation (plamn's watcher writes frames to its stdin):
//  each frame's points (cell centres, every level, finer ones smaller) coloured by value,
//  extremes opaque, the middle of the range faint, so the inside of a 3D cloud shows.
//
// frame (little endian, repeated until EOF; the latest one is drawn):
//   char[4]  "PLMF"
//   u32      title length, then the title's bytes
//   f64      t
//   u64      step
//   u64      n points
//   f32[3n]  x y z per point
//   f32[n]   value per point
//   f32[n]   radius per point (half its level's cell width)
// and optionally, after it, the frame's arrows (a vector field averaged over blocks of cells):
//   char[4]  "PLMV"
//   u64      n arrows
//   f32[3n]  x y z per arrow (its block's centre)
//   f32[3n]  the vector
//   f32[n]   half the block's width (slices: an arrow is in a plane its block crosses)
// After EOF the last frame stays until the window is closed.
//
// moving around: the mouse as mathplot's (primary drag rotates, secondary drag translates, the
//  wheel zooms; Ctrl-h prints its keys), and first person keys: W/S forward / back, A/D left /
//  right, R/F up / down (a step: 2% of the data's extent; hold Shift for 10x), arrows turn the
//  view (3 degrees), Q/E roll. Ctrl-a resets the view.
//
// slices (one cell thick at each point's level: lo <= x - pos < hi, a level's cell centres hit
//  once): 1/2/3 the plane normal to x/y/z, 4/5/6 two of them (xy, xz, yz), 7 all three, 8/9
//  move them down / up a finest cell, 0 the whole cloud (and the planes back to the middle).
//  Slices draw every point in them (the instance cap only bites the whole cloud).
//
// arrows: mathplot's QuiverVisual over the balls, length by magnitude (the longest a block's
//  width), colour by magnitude; filtered by the slices too.
//
// B: the balls shown / hidden, V: the arrows shown / hidden (kept over new frames).
//
// P: play / pause, N: one more frame while paused. With --control-fd N (the watcher's pipe) the
//  viewer writes "pause" / "play" / "step" lines there and the simulation waits while paused;
//  without one, pausing just keeps the current frame.

#include <GLFW/glfw3.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <format>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>

#include <stdio.h> // fdopen
#include <variant>
#include <vector>

import mplot.colourmap;
import mplot.visual;
import mplot.instancedscattervisual;
import mplot.quivervisual;
import mplot.keys;

constexpr int glver = mplot::gl::version_4_3; // instancing: SSBOs

struct Frame
{
    std::string title;
    double t           = 0;
    std::uint64_t step = 0;
    std::vector<float> xyz, value, radius;
};

struct Arrows
{
    std::vector<float> xyz, vec, half;
};

// mathplot's Visual with first person keys (its own keys are Ctrl-something, kept)
struct Viewer : mplot::Visual<glver>
{
    using mplot::Visual<glver>::Visual;
    float step = 0.1f; // scene units per W/A/S/D/R/F press, from the data's extent

    // slices: the planes shown (bit c: normal to axis c, 0: the whole cloud), where, and how
    //  far 8/9 move them (a finest cell); dirty: redraw the current frame
    bool paused = false, step_once = false;
    bool balls = true, arrows = true; // B, V: shown
    std::FILE* control = nullptr;     // the watcher's pipe (--control-fd), if any
    void tell(char const* const what)
    {
        if (control)
        {
            std::fputs(what, control);
            std::fputc('\n', control);
            std::fflush(control);
        }
    }
    unsigned planes = 0;
    std::array<float, 3> centre{}, pos{};
    float finest = 1.f;
    bool dirty   = false;

    // the camera moved by d in the screen frame, as mathplot's secondary-button drag does
    void move(sm::vec<float, 3> const d)
    {
        for (auto* m : {&this->sceneview, &this->sceneview_tr, &this->savedSceneview,
                        &this->savedSceneview_tr, &this->lastSceneview, &this->lastSceneview_tr})
            m->pretranslate(d);
    }
    // the view turned about the camera
    void turn(sm::vec<float, 3> const axis, float const degrees)
    {
        float const theta = degrees * 3.14159265f / 180.f;
        for (auto* m : {&this->sceneview, &this->savedSceneview, &this->lastSceneview})
            m->prerotate(axis, theta);
    }

    void key_callback_extra(std::int32_t const key, std::int32_t, std::int32_t const action,
                            std::int32_t const mods) override
    {
        using k = mplot::key;
        if ((action != mplot::keyaction::press and action != mplot::keyaction::repeat)
            or (mods & mplot::keymod::control))
            return;
        float const d = step * ((mods & mplot::keymod::shift) ? 10.f : 1.f);
        if (mods & mplot::keymod::shift and (key == k::left or key == k::right))
            return; // mathplot's: the selected model's opacity
        if ((key == k::b or key == k::v) and action == mplot::keyaction::press)
        {
            (key == k::b ? balls : arrows) ^= true;
            dirty = true;
            return;
        }
        if (key == k::p and action == mplot::keyaction::press)
        {
            paused = !paused;
            tell(paused ? "pause" : "play");
            return;
        }
        if (key == k::n and paused)
        {
            step_once = true;
            tell("step");
            return;
        }
        static constexpr std::array<unsigned, 8> masks{0, 1, 2, 4, 1 | 2, 1 | 4, 2 | 4, 1 | 2 | 4};
        if (key >= k::n0 and key <= k::n7)
        {
            planes = masks[static_cast<std::size_t>(key - k::n0)];
            if (key == k::n0)
                pos = centre;
            dirty = true;
            return;
        }
        if ((key == k::n8 or key == k::n9) and planes)
        {
            for (std::size_t c = 0; c < 3; ++c)
                if (planes & (1u << c))
                    pos[c] += key == k::n9 ? finest : -finest;
            dirty = true;
            return;
        }
        switch (key)
        {
            case k::w: move({0, 0, d}); break;
            case k::s: move({0, 0, -d}); break;
            case k::a: move({d, 0, 0}); break;
            case k::d: move({-d, 0, 0}); break;
            case k::r: move({0, -d, 0}); break;
            case k::f: move({0, d, 0}); break;
            case k::left: turn({0, 1, 0}, -3); break;
            case k::right: turn({0, 1, 0}, 3); break;
            case k::up: turn({1, 0, 0}, -3); break;
            case k::down: turn({1, 0, 0}, 3); break;
            case k::q: turn({0, 0, 1}, -3); break;
            case k::e: turn({0, 0, 1}, 3); break;
            default: break;
        }
    }
};

namespace
{
template<typename T>
bool read(std::FILE* in, T* out, std::size_t const n = 1)
{
    return n == 0 or std::fread(out, sizeof(T), n, in) == n;
}

std::optional<Arrows> read_arrows(std::FILE* in)
{
    Arrows a;
    std::uint64_t n = 0;
    if (!read(in, &n))
        return std::nullopt;
    a.xyz.resize(3 * n);
    a.vec.resize(3 * n);
    a.half.resize(n);
    if (!read(in, a.xyz.data(), 3 * n) or !read(in, a.vec.data(), 3 * n)
        or !read(in, a.half.data(), n))
        return std::nullopt;
    return a;
}

// the next frame or arrows (a "PLMF" or "PLMV" record); nothing at EOF
std::optional<std::variant<Frame, Arrows>> read_record(std::FILE* in)
{
    std::array<char, 4> magic{};
    if (!read(in, magic.data(), 4))
        return std::nullopt; // EOF
    if (magic == std::array<char, 4>{'P', 'L', 'M', 'V'})
    {
        if (auto a = read_arrows(in))
            return std::variant<Frame, Arrows>{std::move(*a)};
        return std::nullopt;
    }
    if (magic != std::array<char, 4>{'P', 'L', 'M', 'F'})
        throw std::runtime_error("view3d: not a frame (magic)");
    Frame f;
    std::uint32_t len = 0;
    std::uint64_t n   = 0;
    if (!read(in, &len))
        return std::nullopt;
    f.title.resize(len);
    if (!read(in, f.title.data(), len) or !read(in, &f.t) or !read(in, &f.step) or !read(in, &n))
        return std::nullopt;
    f.xyz.resize(3 * n);
    f.value.resize(n);
    f.radius.resize(n);
    if (!read(in, f.xyz.data(), 3 * n) or !read(in, f.value.data(), n)
        or !read(in, f.radius.data(), n))
        return std::nullopt;
    return std::variant<Frame, Arrows>{std::move(f)};
}
} // namespace

int main(int argc, char** argv)
{
    int control_fd = -1;
    for (int i = 1; i + 1 < argc; ++i)
        if (std::string{argv[i]} == "--control-fd")
            control_fd = std::stoi(argv[i + 1]);

    std::mutex mutex;
    std::optional<Frame> latest;         // the reader's newest, taken by the drawing loop
    std::optional<Arrows> latest_arrows; // a frame's arrows: always taken (they come with it,
                                         //  before the watcher waits for P)
    std::atomic<bool> eof{false};
    std::thread reader{[&] {
        try
        {
            while (auto r = read_record(stdin))
            {
                std::lock_guard lock{mutex};
                if (auto* f = std::get_if<Frame>(&*r))
                    latest = std::move(*f);
                else
                    latest_arrows = std::move(std::get<Arrows>(*r));
            }
        }
        catch (std::exception const& e)
        {
            std::cerr << e.what() << "\n";
        }
        eof = true;
    }};
    reader.detach(); // blocked on stdin when the window closes first: let it go with the process

    Viewer v(1280, 900, "plamn view3d");
    if (control_fd >= 0)
        v.control = fdopen(control_fd, "w");
    v.lightingEffects();
    v.showCoordArrows(true);
    v.bgcolour = {0.08f, 0.08f, 0.1f, 1.0f};

    auto model = std::make_unique<mplot::InstancedScatterVisual<glver>>(sm::vec<>{});
    model->set_parent(v.get_id());
    model->radiusFixed   = 1.0f;        // each point's own radius as its scale
    model->max_instances = 32u * 1024u; // all of mathplot's instance storage (one model here)
    model->finalize();
    auto* points = v.addVisualModel(model);
    v.render(); // the model's instance storage is set up at its first render (as mathplot's
                //  scatter_instanced example: set_instance_data after it)
    std::size_t const capacity = points->max_instances; // mathplot's SSBO share, 16k by default

    mplot::ColourMap<float> cm{mplot::ColourMapType::Plasma};
    sm::vvec<sm::vec<float, 3>> xyz;
    sm::vvec<std::array<float, 3>> colours;
    sm::vvec<float> alpha, scale;

    std::optional<Frame> current; // the frame drawn: redrawn on a key (slices) too
    std::optional<Arrows> current_arrows;
    mplot::VisualModel<glver>* quiver = nullptr; // the arrows' model, remade with them
    std::vector<sm::vec<float>> qcoords, qvecs;  // its data (it keeps pointers to them)
    bool first = true;
    while (!v.readyToFinish())
    {
        bool fresh = false, fresh_arrows = false;
        {
            std::lock_guard lock{mutex};
            if (latest and (!v.paused or v.step_once))
            {
                v.step_once = false;
                current.swap(latest);
                latest.reset();
                fresh = true;
            }
            if (latest_arrows)
            {
                current_arrows.swap(latest_arrows);
                latest_arrows.reset();
                fresh_arrows = true;
            }
        }
        bool const dirty = v.dirty;
        v.dirty          = false;
        if (current_arrows and (fresh_arrows or dirty))
        {
            // the arrows in the shown planes (their block crosses one), or all of them
            auto const& a = *current_arrows;
            qcoords.clear();
            qvecs.clear();
            float longest = 0;
            for (std::size_t i = 0; i < a.half.size(); ++i)
            {
                bool in = v.planes == 0;
                for (std::size_t c = 0; c < 3 and !in; ++c)
                    in = (v.planes & (1u << c))
                         and std::abs(a.xyz[3 * i + c] - v.pos[c]) < a.half[i];
                if (!in)
                    continue;
                qcoords.push_back({a.xyz[3 * i], a.xyz[3 * i + 1], a.xyz[3 * i + 2]});
                qvecs.push_back({a.vec[3 * i], a.vec[3 * i + 1], a.vec[3 * i + 2]});
                longest = std::max(longest, 2 * a.half[i]);
            }
            if (quiver)
                v.removeVisualModel(quiver);
            quiver = nullptr;
            if (!qcoords.empty() and v.arrows)
            {
                auto q = std::make_unique<mplot::QuiverVisual<float, glver>>(
                    &qcoords, sm::vec<float>{}, &qvecs, mplot::ColourMapType::Greyscale);
                q->set_parent(v.get_id());
                q->quiver_length_gain    = longest; // the longest arrow: a block's width
                q->quiver_thickness_gain = 0.05f;
                q->finalize();
                quiver = v.addVisualModel(q);
            }
        }
        if (current and (fresh or dirty))
        {
            auto const& f    = *current;
            auto const total = f.value.size();
            float lo = 0, hi = 1;
            if (total)
            {
                auto const [a, b] = std::minmax_element(f.value.begin(), f.value.end());
                lo                = *a;
                hi                = *b > *a ? *b : *a + 1;
            }
            if (fresh and total) // extent: the key step, the slices' middle, a finest cell
            {
                float ext = 0;
                for (std::size_t c = 0; c < 3; ++c)
                {
                    float a = f.xyz[c], b = a;
                    for (std::size_t i = 0; i < total; ++i)
                        a = std::min(a, f.xyz[3 * i + c]), b = std::max(b, f.xyz[3 * i + c]);
                    ext         = std::max(ext, b - a);
                    v.centre[c] = (a + b) / 2;
                }
                v.step   = 0.02f * std::max(ext, 1e-6f);
                v.finest = 2 * *std::min_element(f.radius.begin(), f.radius.end());
                if (first)
                    v.pos = v.centre, first = false;
            }

            // the points drawn: those in a shown plane (one cell thick at their level), or
            //  every stride-th of the whole cloud if more than the model holds
            std::vector<std::size_t> pick;
            if (v.planes)
            {
                for (std::size_t i = 0; i < total; ++i)
                    for (std::size_t c = 0; c < 3; ++c)
                    {
                        float const d = f.xyz[3 * i + c] - v.pos[c];
                        if ((v.planes & (1u << c)) and -f.radius[i] <= d and d < f.radius[i])
                        {
                            pick.push_back(i);
                            break;
                        }
                    }
            }
            else
                for (std::size_t i = 0; i < total; ++i)
                    pick.push_back(i);
            auto const shown  = pick.size();
            auto const stride = std::max<std::size_t>(1, (shown + capacity - 1) / capacity);
            auto const n      = (shown + stride - 1) / stride;
            xyz.resize(n);
            colours.resize(n);
            alpha.resize(n);
            scale.resize(n);
            for (std::size_t k = 0; k < n; ++k)
            {
                auto const i  = pick[k * stride];
                xyz[k]        = {f.xyz[3 * i], f.xyz[3 * i + 1], f.xyz[3 * i + 2]};
                float const u = (f.value[i] - lo) / (hi - lo); // 0 .. 1
                colours[k]    = cm.convert(u);
                alpha[k]      = v.planes ? 1.f : 0.05f + 0.95f * std::abs(2 * u - 1);
                scale[k]      = f.radius[i];
            }
            if (!v.balls) // hidden: none drawn (the title still counts them)
                xyz.clear(), colours.clear(), alpha.clear(), scale.clear();
            points->set_instance_data(xyz, colours, alpha, scale);

            std::string where;
            for (std::size_t c = 0; c < 3; ++c)
                if (v.planes & (1u << c))
                    where += std::format("  {}={:.4g}", "xyz"[c], v.pos[c]);
            auto const title = std::format(
                "{}  t = {:.6g}  step {}  [{:.4g}, {:.4g}]{}  {} of {} points{}", f.title, f.t,
                f.step, lo, hi, where.empty() ? std::string{} : "  slice" + where, n, total,
                eof        ? "  (ended)"
                : v.paused ? "  (paused: P plays, N steps)"
                           : "");
            glfwSetWindowTitle(v.getWindow(), title.c_str());
        }
        v.render();
        v.poll();
        std::this_thread::sleep_for(std::chrono::milliseconds(16));
    }
    return 0;
}
