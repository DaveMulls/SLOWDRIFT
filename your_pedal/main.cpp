// SlowDrift - full C++ port of the Pure Data patch (SD_LowLatency_v7.pd)
// 48 kHz, block size 2. Controls read in the main loop.
//
// The DSP primitives below deliberately reproduce Pure Data's exact
// coefficient maths (lop~, hip~, bp~, rpole~, vd~) so the voicing matches the
// patch rather than merely resembling it.
//
// CONTROLS
//   knob1  output level        0..4x
//   knob2  modulation depth    wow / vibrato / tremolo
//   knob3  modulation rate     + noise-wow amount
//   knob4  drive               1..31x into the saturator
//   knob5  flavour mode        6 positions
//   knob6  failure amount      dropouts, snags, bloom
//   sw1    dry blend on
//   sw2    compressor makeup
//   sw3    alt EQ (removes the pre-saturation dry path)
//   sw4    vibrato mode (replaces wow with a clean LFO)
//   fs1    effect on/off (latching)
//   fs2    tape stop (hold)

#include "daisy_seed.h"
#include "daisysp.h"
#include <cmath>

using namespace daisy;
using namespace daisysp;

// ===========================================================================
// Pd-faithful DSP primitives
// ===========================================================================

static inline float Clampf(float x, float lo, float hi)
{
    return x < lo ? lo : (x > hi ? hi : x);
}

// heavylib hv.tanh~ equivalent: Pade rational approximation of tanh
static inline float FastTanh(float x)
{
    if(x < -3.f)
        return -1.f;
    if(x > 3.f)
        return 1.f;
    const float x2 = x * x;
    return x * (27.f + x2) / (27.f + 9.f * x2);
}

// Pd [lop~] : coef = clamp(2*pi*fc/sr, 0, 1); y += coef * (x - y)
struct Lop
{
    float coef = 0.f, z = 0.f, sr = 48000.f;
    void  Init(float sampleRate, float fc)
    {
        sr = sampleRate;
        SetFreq(fc);
        z = 0.f;
    }
    void SetFreq(float fc) { coef = Clampf(2.f * (float)M_PI * fc / sr, 0.f, 1.f); }
    inline float Process(float x)
    {
        z += coef * (x - z);
        return z;
    }
};

// Pd [hip~] response, computed as x - lowpass(x).
// Mathematically the same one-pole high pass, but without the huge internal
// accumulator that costs precision in 32-bit float at low cutoffs.
struct Hip
{
    Lop  lp;
    void Init(float sr, float fc) { lp.Init(sr, fc); }
    void SetFreq(float fc) { lp.SetFreq(fc); }
    inline float Process(float x) { return x - lp.Process(x); }
};

// Pd [bp~] : two-pole resonator
struct Bp
{
    float coef1 = 0.f, coef2 = 0.f, gain = 0.f;
    float last = 0.f, prev = 0.f, sr = 48000.f;

    static inline float QCos(float f)
    {
        if(f >= -1.5707963f && f <= 1.5707963f)
        {
            const float g = f * f;
            return (((g * g * g * (-1.f / 720.f) + g * g * (1.f / 24.f)) - g * 0.5f) + 1.f);
        }
        return 0.f;
    }
    void Init(float sampleRate, float f, float q)
    {
        sr   = sampleRate;
        last = prev = 0.f;
        Set(f, q);
    }
    void Set(float f, float q)
    {
        if(f < 0.001f)
            f = 10.f;
        if(q < 0.f)
            q = 0.f;
        const float omega      = f * (2.f * (float)M_PI) / sr;
        float       oneminusr  = (q < 0.001f) ? 1.f : omega / q;
        if(oneminusr > 1.f)
            oneminusr = 1.f;
        const float r = 1.f - oneminusr;
        coef1         = 2.f * QCos(omega) * r;
        coef2         = -r * r;
        gain          = 2.f * oneminusr * (oneminusr + r * omega);
    }
    inline float Process(float x)
    {
        const float output = x + coef1 * last + coef2 * prev;
        prev               = last;
        last               = output;
        return gain * output;
    }
};

// Pd [rpole~] : y[n] = x[n] + a * y[n-1]
struct RPole
{
    float y = 0.f;
    inline float Process(float x, float a)
    {
        y = x + a * y;
        return y;
    }
};

// Pd [line~] : linear ramp to a target over a time in ms
struct Line
{
    float value = 0.f, target = 0.f, inc = 0.f, sr = 48000.f;
    int   samplesLeft = 0;
    void  Init(float sampleRate, float v = 0.f)
    {
        sr = sampleRate;
        value = target = v;
        samplesLeft    = 0;
    }
    void Set(float t, float ms)
    {
        target = t;
        if(ms <= 0.f)
        {
            value       = t;
            samplesLeft = 0;
            inc         = 0.f;
            return;
        }
        samplesLeft = (int)(ms * 0.001f * sr);
        if(samplesLeft < 1)
            samplesLeft = 1;
        inc = (target - value) / (float)samplesLeft;
    }
    inline float Process()
    {
        if(samplesLeft > 0)
        {
            value += inc;
            if(--samplesLeft == 0)
                value = target;
        }
        return value;
    }
};

// Pd [phasor~] : 0..1 ramp
struct Phasor
{
    float phase = 0.f, inc = 0.f, sr = 48000.f;
    void  Init(float sampleRate) { sr = sampleRate; }
    void  SetFreq(float f) { inc = f / sr; }
    void  SetPhase(float p) { phase = p; }
    inline float Process()
    {
        phase += inc;
        while(phase >= 1.f)
            phase -= 1.f;
        while(phase < 0.f)
            phase += 1.f;
        return phase;
    }
};

// Pd [delwrite~] / [vd~] with 4-point cubic interpolation, as vd~ uses.
template <size_t N>
struct Delay
{
    float  buf[N];
    size_t wr = 0;
    float  sr = 48000.f;

    void Init(float sampleRate)
    {
        sr = sampleRate;
        for(size_t i = 0; i < N; i++)
            buf[i] = 0.f;
        wr = 0;
    }
    inline void Write(float x)
    {
        buf[wr] = x;
        wr      = (wr + 1) % N;
    }
    inline float ReadMs(float ms)
    {
        float d = ms * 0.001f * sr;
        if(d < 1.f)
            d = 1.f;
        if(d > (float)(N - 3))
            d = (float)(N - 3);

        const int   di   = (int)d;
        const float frac = d - (float)di;

        const size_t base = (wr + N - (size_t)di) % N;
        const float  a    = buf[(base + N - 2) % N];
        const float  b    = buf[(base + N - 1) % N];
        const float  c    = buf[base % N];
        const float  dd   = buf[(base + 1) % N];

        // Pd's 4-point cubic
        const float cminusb = c - b;
        return b
               + frac
                     * (cminusb
                        - 0.1666667f * (1.f - frac)
                              * ((dd - a - 3.f * cminusb) * frac
                                 + (dd + 2.f * a - 3.f * b)));
    }
};

// Pd [noise~]
struct Noise
{
    uint32_t state = 22222;
    inline float Process()
    {
        state = state * 1664525u + 1013904223u;
        return (float)((int32_t)state) * 4.6566129e-10f;
    }
};

// ===========================================================================
// Control block, written by the main loop
// ===========================================================================
struct Controls
{
    float knob[6]  = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    bool  dryOn    = false; // sw1
    bool  compOn   = false; // sw2
    bool  altEq    = false; // sw3
    bool  vibrato  = false; // sw4
    bool  effectOn = false; // fs1, latching
    bool  tapeStop = false; // fs2, held
};

// ===========================================================================
// The pedal. No libDaisy types, so this also builds in a desktop WAV harness.
// ===========================================================================
struct SlowDrift
{
    float sr = 48000.f;

    // --- input / output conditioning
    Hip inHp, outHp, preHp50, satHp1, satHp2;

    // --- bloom (lpg_engine)
    Hip   lpgHp;
    Lop   lpgAtk, lpgRel, lpgRipple, lpgKnob;
    RPole lpgP1, lpgP2;

    // --- saturation and compressor
    Lop   driveSm;
    float compEnvSq = 0.f;
    float compGain  = 1.f;

    // --- tape delay and wow
    Delay<6200> tape;
    Noise       noise;
    Lop         wowN1, wowN2, wowN3;   // noise~ -> lop 0.8 / 15 / 8
    Line        walkLine;              // random-walk target ramp
    Lop         walkS1, walkS2, walkHp;
    Phasor      sine1, sine2, sine3;   // 0.61 / 1.07 / 1.83 Hz
    Phasor      vibLfo;
    Lop         vibDepth, altDepth, snagSm1, snagSm2;
    float       walkTarget = 0.f;
    int         walkTimer  = 0;
    float       snagTime   = 2.f;
    int         snagTimer  = -1;

    // --- failure engine
    int   failTimer     = 0;
    float dropoutTarget = 1.f;
    Lop   dropoutSm;

    // --- flavour bus
    Line  modeGain[6];
    int   currentMode = -1;
    // mode 1: harmonic
    Lop   f1a, f1b;
    Bp    f1bp;
    // mode 2: alt EQ
    Hip   f2h1, f2h2;
    Lop   f2l1, f2l2;
    Bp    f2bp;
    // mode 3: tape flange
    Delay<800>  flange;
    Lop         flangeMod1, flangeMod2;
    // mode 4: tremolo
    Phasor tremLfo;
    Lop    tremDepth;
    // mode 5: slap
    Delay<7400> slap;
    Lop         slapLp;

    // --- tape stop
    Delay<29000> tsTape;
    Delay<6200>  tsPitch;
    Line         tsSweep, tsTapeGain, tsPitchRate, tsPitchGain;
    Phasor       tsGrain1, tsGrain2;
    bool         tsPrev = false;
    int          tsStage = -1, tsTimer = 0;

    uint32_t rng = 12345;
    inline float Rand01()
    {
        rng = rng * 1664525u + 1013904223u;
        return (float)(rng >> 8) * (1.f / 16777216.f);
    }

    void Init(float sampleRate)
    {
        sr = sampleRate;

        inHp.Init(sr, 10.f);
        outHp.Init(sr, 10.f);
        preHp50.Init(sr, 50.f);
        satHp1.Init(sr, 30.f);
        satHp2.Init(sr, 10.f);

        lpgHp.Init(sr, 10.f);
        lpgAtk.Init(sr, 80.f);
        lpgRel.Init(sr, 3.f);
        lpgRipple.Init(sr, 20.f);
        lpgKnob.Init(sr, 10.f);

        driveSm.Init(sr, 5.f);

        tape.Init(sr);
        wowN1.Init(sr, 0.8f);
        wowN2.Init(sr, 15.f);
        wowN3.Init(sr, 8.f);
        walkLine.Init(sr, 0.f);
        walkS1.Init(sr, 0.5f);
        walkS2.Init(sr, 0.5f);
        walkHp.Init(sr, 0.3f);
        sine1.Init(sr);
        sine1.SetFreq(0.61f);
        sine2.Init(sr);
        sine2.SetFreq(1.07f);
        sine2.SetPhase(0.37f);
        sine3.Init(sr);
        sine3.SetFreq(1.83f);
        sine3.SetPhase(0.71f);
        vibLfo.Init(sr);
        vibDepth.Init(sr, 10.f);
        altDepth.Init(sr, 20.f);
        snagSm1.Init(sr, 1.f);
        snagSm2.Init(sr, 1.f);
        dropoutSm.Init(sr, 10.f);

        for(int i = 0; i < 6; i++)
            modeGain[i].Init(sr, i == 0 ? 1.f : 0.f);

        f1a.Init(sr, 1000.f);
        f1b.Init(sr, 500.f);
        f1bp.Init(sr, 600.f, 10.f);
        f2h1.Init(sr, 500.f);
        f2h2.Init(sr, 500.f);
        f2l1.Init(sr, 1800.f);
        f2l2.Init(sr, 1800.f);
        f2bp.Init(sr, 1000.f, 0.2f);
        flange.Init(sr);
        flangeMod1.Init(sr, 0.3f);
        flangeMod2.Init(sr, 0.3f);
        tremLfo.Init(sr);
        tremDepth.Init(sr, 10.f);
        slap.Init(sr);
        slapLp.Init(sr, 4000.f);

        tsTape.Init(sr);
        tsPitch.Init(sr);
        tsSweep.Init(sr, 0.f);
        tsTapeGain.Init(sr, 1.f);
        tsPitchRate.Init(sr, 0.001f);
        tsPitchGain.Init(sr, 0.f);
        tsGrain1.Init(sr);
        tsGrain2.Init(sr);
        tsGrain2.SetPhase(0.5f);
    }

    // Control-rate housekeeping. Called once per audio block, not per sample.
    void UpdateControls(const Controls& c)
    {
        // knob5 selects one of six flavour modes with a 0.5 ms crossfade
        int mode = (int)(c.knob[4] * 6.f);
        if(mode > 5)
            mode = 5;
        if(mode != currentMode)
        {
            currentMode = mode;
            for(int i = 0; i < 6; i++)
                modeGain[i].Set(i == mode ? 1.f : 0.f, 0.5f);
        }

        // LFO rates follow knob3
        vibLfo.SetFreq(0.15f + c.knob[2] * 6.8f);
        tremLfo.SetFreq(0.5f + c.knob[2] * 14.5f);

        // --- failure engine, driven by knob6 -----------------------------
        // metro period 50..500 ms, shorter as the knob rises
        const float periodMs = 50.f + (1.f - c.knob[5]) * 450.f;
        if(--failTimer <= 0)
        {
            failTimer = (int)(periodMs * 0.001f * sr);
            if(c.knob[5] > 0.001f)
            {
                const float roll = Rand01();
                if(roll < c.knob[5] * 0.9f)
                    dropoutTarget = 0.1f; // dropout
                else
                    dropoutTarget = 1.f;
                if(roll < c.knob[5] * 0.7f && snagTimer < 0)
                    snagTimer = (int)(0.300f * sr); // snag in 300 ms
            }
            else
            {
                dropoutTarget = 1.f;
            }
        }
        if(snagTimer >= 0 && --snagTimer == 0)
        {
            snagTime  = 2.f + c.knob[5] * 20.f;
            snagTimer = -(int)(0.200f * sr); // hold, then release below
        }
        if(snagTimer < -1)
        {
            if(++snagTimer == -1)
                snagTime = 2.f;
        }

        // --- tape stop state machine -------------------------------------
        if(c.tapeStop != tsPrev)
        {
            tsPrev = c.tapeStop;
            if(c.tapeStop)
            {
                tsSweep.Set(1.f, 1000.f);      // read pointer falls back
                tsTapeGain.Set(0.f, 1000.f);   // tape voice fades out
                tsPitchGain.Set(0.f, 10.f);
                tsPitchRate.Set(5.f, 1.f);
                tsStage = 0;
                tsTimer = (int)(0.010f * sr);
            }
            else
            {
                tsSweep.Set(0.f, 30.f);
                tsTapeGain.Set(1.f, 200.f);
                tsPitchGain.Set(0.f, 200.f);
                tsStage = -1;
            }
        }
        if(tsStage == 0 && --tsTimer <= 0)
        {
            tsPitchRate.Set(0.001f, 1000.f); // grain rate collapses
            tsPitchGain.Set(0.8f, 40.f);
            tsStage = 1;
            tsTimer = (int)(1.100f * sr);
        }
        else if(tsStage == 1 && --tsTimer <= 0)
        {
            tsPitchGain.Set(0.f, 200.f);
            tsStage = -1;
        }
    }

    inline float Process(float in, const Controls& c)
    {
        const float dry = inHp.Process(in);

        if(!c.effectOn)
            return Clampf(dry, -0.99f, 0.99f);

        // ---------------- bloom / LPG (vactrol model) --------------------
        const float lpgIn  = lpgHp.Process(dry);
        const float rect   = fabsf(lpgIn);
        const float atk    = lpgAtk.Process(rect);
        const float rel    = lpgRel.Process(rect);
        const float env    = lpgRipple.Process(atk > rel ? atk : rel);
        const float knob   = lpgKnob.Process(c.knob[5]);
        const float open_  = Clampf(Clampf(env * 6.f, 0.f, 1.f) * knob + knob * knob, 0.f, 1.f);
        const float o2     = open_ * open_;
        const float cutoff = o2 * 0.925f + 0.025f;
        const float fb     = 1.f - cutoff;
        float       lpg    = lpgP1.Process(lpgIn * cutoff, fb);
        lpg                = lpgP2.Process(lpg * cutoff, fb);
        lpg *= (open_ * 0.65f + 0.35f);

        // ---------------- pre-saturation sum -----------------------------
        // bloom is summed with the dry path, then high-passed at 50 Hz
        const float x = preHp50.Process(lpg + (c.altEq ? 0.f : dry));

        // ---------------- drive + saturation -----------------------------
        const float drive = driveSm.Process(1.f + c.knob[3] * 30.f);
        float       sat   = FastTanh(x * drive);
        sat               = satHp1.Process(sat);

        // ---------------- compressor (env~ 256 style, ratio 80) ----------
        compEnvSq += (sat * sat - compEnvSq) * 0.004f; // ~256-sample window
        const float rms  = sqrtf(compEnvSq) + 1e-9f;
        const float thr  = 0.1f;
        float       want = 1.f;
        if(rms > thr)
            want = thr / rms; // ratio 80:1 is effectively limiting
        compGain += (want - compGain) * 0.01f;
        float comp = sat * compGain * (c.compOn ? 2.f : 1.f);
        comp       = FastTanh(comp);
        comp       = satHp2.Process(comp);

        // ---------------- wow / vibrato modulation -----------------------
        // noise branch
        float n = wowN3.Process(wowN2.Process(wowN1.Process(noise.Process()))) * 1000.f;
        n       = Clampf(n, -18.f, 18.f);
        const float k3   = c.knob[2] / 3.f;
        const float wowN = (n + c.knob[2] * 6.f) * k3;

        // random-walk branch
        if(--walkTimer <= 0)
        {
            walkTimer = (int)((0.500f + Rand01()) * sr);
            walkTarget = Rand01() * 2.f - 1.f;
            walkLine.Set(walkTarget, 800.f + c.knob[1] * 1200.f);
        }
        float w = walkS2.Process(walkS1.Process(walkLine.Process()));
        w       = (w - walkHp.Process(w)) * 1.05f;

        // sine trio
        const float s = (cosf(6.2831853f * sine1.Process())
                         + cosf(6.2831853f * sine2.Process())
                         + cosf(6.2831853f * sine3.Process()))
                        * 0.221f;

        const float depth = altDepth.Process(c.knob[1]);
        const float mod   = Clampf(w + s, -1.f, 1.f);
        const float wowA  = mod * depth * 5.f + depth * 5.f;

        // vibrato branch (replaces wow when sw4 is up)
        const float vdep = vibDepth.Process(c.knob[1] * 3.f);
        const float vib  = cosf(6.2831853f * vibLfo.Process()) * vdep;

        const float snag = snagSm2.Process(snagSm1.Process(snagTime));

        float delayMs;
        if(c.vibrato)
            delayMs = 4.f + vib + snag;
        else
            delayMs = wowN + wowA + snag;
        delayMs = Clampf(delayMs, 1.f, 120.f);

        tape.Write(comp);
        float t = tape.ReadMs(delayMs) * 0.5f;

        // ---------------- dropout ---------------------------------------
        t *= dropoutSm.Process(dropoutTarget);
        t *= 0.5f;

        // ---------------- flavour bus ------------------------------------
        const float f = t;
        float       bus = 0.f;

        const float g0 = modeGain[0].Process();
        const float g1 = modeGain[1].Process();
        const float g2 = modeGain[2].Process();
        const float g3 = modeGain[3].Process();
        const float g4 = modeGain[4].Process();
        const float g5 = modeGain[5].Process();

        // 0 - clean
        bus += f * g0;

        // 1 - harmonic saturator
        {
            const float d  = FastTanh(f * 1.5f);
            const float lo = f1b.Process(f1a.Process(d)) * 2.f;
            const float hi = FastTanh(f1bp.Process(d) * 10.f);
            bus += (lo + hi) * 0.2f * g1;
        }

        // 2 - alt EQ
        {
            const float band = f2l2.Process(f2l1.Process(f2h2.Process(f2h1.Process(f))));
            const float peak = f2bp.Process(band) * 6.f;
            bus += (band + peak) * 0.5f * g2;
        }

        // 3 - tape flange
        {
            const float mt = flangeMod2.Process(flangeMod1.Process(noise.Process()));
            const float fd = 7.f + mt * 5.f;
            const float fv = flange.ReadMs(Clampf(fd, 1.f, 14.f));
            flange.Write(f + fv * 0.2f);
            bus += (f + fv) * 0.6f * g3;
        }

        // 4 - tremolo
        {
            const float td = tremDepth.Process(c.knob[1] * 0.5f);
            const float lf = cosf(6.2831853f * tremLfo.Process()) * td + (1.f - td);
            bus += f * lf * g4;
        }

        // 5 - slap
        {
            slap.Write(f);
            const float sv = slapLp.Process(slap.ReadMs(120.f)) * 0.5f;
            bus += (f + sv) * g5;
        }

        // ---------------- tape stop --------------------------------------
        tsTape.Write(bus);
        tsPitch.Write(bus);

        const float sweep = tsSweep.Process();
        const float tsDly = sweep * sweep * 490.f + 2.f;
        float       stopped = tsTape.ReadMs(tsDly) * tsTapeGain.Process();

        const float grate = tsPitchRate.Process();
        tsGrain1.SetFreq(grate);
        tsGrain2.SetFreq(grate);
        const float p1 = tsGrain1.Process();
        const float p2 = tsGrain2.Process();
        const float w1 = cosf(6.2831853f * p1) * -0.5f + 0.5f;
        const float w2 = cosf(6.2831853f * p1) * 0.5f + 0.5f;
        const float gr = tsPitch.ReadMs(p1 * 100.f + 1.f) * w1
                         + tsPitch.ReadMs(p2 * 100.f + 1.f) * w2;
        stopped += gr * tsPitchGain.Process();

        // ---------------- output -----------------------------------------
        float outv = stopped * (c.knob[0] * 4.f);
        if(c.dryOn)
            outv += comp * 0.35f;

        outv = outHp.Process(outv);
        return Clampf(outv, -0.99f, 0.99f);
    }
};

// ===========================================================================
// Hardware
// ===========================================================================
namespace pins
{
constexpr Pin KNOB[6] = {seed::A0, seed::A1, seed::A2,
                         seed::A3, seed::A4, seed::A5};
constexpr Pin SWITCH[4]     = {seed::D10, seed::D9, seed::D8, seed::D7};
constexpr Pin FOOTSWITCH[2] = {seed::D25, seed::D26};
constexpr Pin LED[2]        = {seed::D22, seed::D23};
} // namespace pins

DaisySeed     hw;
Controls      controls;
SlowDrift     pedal;
AnalogControl knobCtl[6];
Switch        swCtl[4];
Switch        fsCtl[2];
GPIO          ledOut[2];

void AudioCallback(AudioHandle::InterleavingInputBuffer  in,
                   AudioHandle::InterleavingOutputBuffer out,
                   size_t                                size)
{
    pedal.UpdateControls(controls);
    for(size_t i = 0; i < size; i += 2)
    {
        const float s = pedal.Process(in[i], controls);
        out[i]        = s;
        out[i + 1]    = s;
    }
}

int main(void)
{
    hw.Init();
    hw.SetAudioSampleRate(SaiHandle::Config::SampleRate::SAI_48KHZ);
    hw.SetAudioBlockSize(2);

    const float sr = hw.AudioSampleRate();
    pedal.Init(sr);

    AdcChannelConfig adcCfg[6];
    for(int i = 0; i < 6; i++)
        adcCfg[i].InitSingle(pins::KNOB[i]);
    hw.adc.Init(adcCfg, 6);
    hw.adc.Start();

    for(int i = 0; i < 6; i++)
        knobCtl[i].Init(hw.adc.GetPtr(i), 1000.f, false, false, 0.01f);
    for(int i = 0; i < 4; i++)
        swCtl[i].Init(pins::SWITCH[i], 1000.f);
    for(int i = 0; i < 2; i++)
        fsCtl[i].Init(pins::FOOTSWITCH[i], 1000.f);
    for(int i = 0; i < 2; i++)
        ledOut[i].Init(pins::LED[i], GPIO::Mode::OUTPUT);

    hw.StartAudio(AudioCallback);

    for(;;)
    {
        for(int i = 0; i < 6; i++)
            controls.knob[i] = knobCtl[i].Process();

        for(int i = 0; i < 4; i++)
            swCtl[i].Debounce();
        controls.dryOn   = swCtl[0].Pressed();
        controls.compOn  = swCtl[1].Pressed();
        controls.altEq   = swCtl[2].Pressed();
        controls.vibrato = swCtl[3].Pressed();

        for(int i = 0; i < 2; i++)
            fsCtl[i].Debounce();
        if(fsCtl[0].RisingEdge())
            controls.effectOn = !controls.effectOn;
        controls.tapeStop = fsCtl[1].Pressed();

        ledOut[0].Write(controls.effectOn);
        ledOut[1].Write(controls.tapeStop);

        System::Delay(1);
    }
}
